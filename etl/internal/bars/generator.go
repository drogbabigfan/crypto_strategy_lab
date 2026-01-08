package bars

import (
	"fmt"
	"time"
)

const (
	// WarmupDays is the number of days needed before bar generation starts.
	// Look-ahead Bias Zero: Today's threshold uses only data from t-14 to t-1.
	WarmupDays = 14

	// DefaultTargetBarsPerDay is the default target number of bars per day (~29 min intervals).
	DefaultTargetBarsPerDay = 50

	// EMAAlpha is the smoothing factor for EMA calculation.
	// Alpha = 2 / (WarmupDays + 1) ≈ 0.133
	// Higher alpha = more weight on recent data.
	EMAAlpha = 2.0 / (WarmupDays + 1)

	// MillisPerDay is milliseconds in a day.
	MillisPerDay = 24 * 60 * 60 * 1000

	// MaxTimestampGapDays is the maximum allowed gap between state timestamp and incoming trade.
	// If a trade's timestamp is more than this many days BEFORE the state's current day,
	// it indicates state corruption (not just resume skip).
	// Set to 45 days to allow for month boundary processing with buffer.
	MaxTimestampGapDays = 45
)

// ErrStateCorruption is returned when state corruption is detected.
type ErrStateCorruption struct {
	Message       string
	StateDay      int64
	TradeDay      int64
	GapDays       int
	LastTradeID   int64
	CurrentTradeID int64
}

func (e *ErrStateCorruption) Error() string {
	return fmt.Sprintf("STATE CORRUPTION DETECTED: %s (state_day=%d, trade_day=%d, gap=%d days, last_trade_id=%d, current_trade_id=%d)",
		e.Message, e.StateDay, e.TradeDay, e.GapDays, e.LastTradeID, e.CurrentTradeID)
}

// Generator creates Adaptive Dynamic Dollar Bars from trade data.
// Uses 14-day EMA of daily volume for threshold calculation.
// Look-ahead Bias Zero: Threshold is updated only at UTC 00:00.
type Generator struct {
	// Configuration
	targetBarsPerDay int // Target number of bars per day (default: 50)

	// Daily volume tracking (circular buffer for warmup SMA initialization)
	dailyVolumes     [WarmupDays]float64 // Daily dollar volumes (used during warmup)
	dailyVolumeHead  int                 // Next write position
	dailyVolumeCount int                 // Number of valid entries (0-14)

	// EMA tracking (used after warmup)
	emaVolume float64 // Exponential Moving Average of daily volume

	// Current day tracking
	currentDayStart  int64   // UTC 00:00 of current day (ms)
	currentDayVolume float64 // Accumulated volume for current day

	// Threshold (updated daily at UTC 00:00)
	currentThreshold float64

	// Current bar accumulator
	acc BarAccumulator

	// Warmup state
	isWarmup bool // true during first 14 days (no bar generation)

	// Resume tracking - prevents double counting on crash recovery
	lastProcessedTradeID int64 // Last successfully processed trade ID (Binance aggTradeId)

	// Monthly checkpoints for completeness verification
	monthlyCheckpoints     map[string]MonthCheckpoint
	currentProcessingMonth string
}

// NewGenerator creates a new bar generator with default settings.
// Starts in warmup mode - first 14 days only collect daily volume data.
func NewGenerator() *Generator {
	return NewGeneratorWithBarsPerDay(DefaultTargetBarsPerDay)
}

// NewGeneratorWithBarsPerDay creates a new bar generator with specified bars per day.
// Starts in warmup mode - first 14 days only collect daily volume data.
func NewGeneratorWithBarsPerDay(barsPerDay int) *Generator {
	if barsPerDay <= 0 {
		barsPerDay = DefaultTargetBarsPerDay
	}
	return &Generator{
		targetBarsPerDay:   barsPerDay,
		isWarmup:           true, // Start in warmup mode
		monthlyCheckpoints: make(map[string]MonthCheckpoint),
	}
}

// ProcessTrade processes a single trade and returns a bar if threshold is reached.
// During warmup period (first 14 days), returns nil and only tracks daily volume.
// Deprecated: Use ProcessTradeWithID for crash recovery support.
func (g *Generator) ProcessTrade(price, quantity, quoteQty float64, timestamp int64, isAggressorBuy bool) *DynamicDollarBar {
	return g.ProcessTradeWithID(0, price, quantity, quoteQty, timestamp, isAggressorBuy)
}

// ProcessTradeWithID processes a single trade with crash recovery support.
// tradeID: Binance trade ID (aggTradeId) for resume tracking.
// If tradeID <= lastProcessedTradeID, the trade is skipped to prevent double counting.
// Returns a bar if threshold is reached, nil otherwise.
//
// Deprecated: Use ProcessTradeWithIDSafe for proper error handling.
func (g *Generator) ProcessTradeWithID(tradeID int64, price, quantity, quoteQty float64, timestamp int64, isAggressorBuy bool) *DynamicDollarBar {
	bar, _ := g.ProcessTradeWithIDSafe(tradeID, price, quantity, quoteQty, timestamp, isAggressorBuy)
	return bar
}

// ProcessTradeWithIDSafe processes a single trade with crash recovery and state corruption detection.
// Returns ErrStateCorruption if the trade timestamp is suspiciously old compared to state.
// This is the recommended method for production use.
func (g *Generator) ProcessTradeWithIDSafe(tradeID int64, price, quantity, quoteQty float64, timestamp int64, isAggressorBuy bool) (*DynamicDollarBar, error) {
	// Fail-Fast: Detect state corruption
	// If trade timestamp is more than MaxTimestampGapDays before state's current day,
	// this indicates we're processing old data with a corrupted (future) state.
	if g.currentDayStart > 0 {
		tradeDayStart := getDayStartUTC(timestamp)
		gapMs := g.currentDayStart - tradeDayStart
		gapDays := int(gapMs / MillisPerDay)

		if gapDays > MaxTimestampGapDays {
			return nil, &ErrStateCorruption{
				Message:        "trade timestamp is too far in the past compared to state",
				StateDay:       g.currentDayStart,
				TradeDay:       tradeDayStart,
				GapDays:        gapDays,
				LastTradeID:    g.lastProcessedTradeID,
				CurrentTradeID: tradeID,
			}
		}
	}

	// Skip already-processed trades (resume protection)
	if tradeID > 0 && tradeID <= g.lastProcessedTradeID {
		return nil, nil // Already processed, skip to prevent double counting
	}

	// Check for day change and update threshold if needed
	g.handleDayChange(timestamp)

	// Accumulate daily volume (always, even during warmup)
	g.currentDayVolume += quoteQty

	// Update last processed trade ID
	if tradeID > g.lastProcessedTradeID {
		g.lastProcessedTradeID = tradeID
	}

	// During warmup, don't generate bars
	if g.isWarmup {
		return nil, nil
	}

	// Accumulate trade into current bar
	g.acc.AddTrade(price, quantity, quoteQty, timestamp, isAggressorBuy)

	// Check if threshold reached
	if g.acc.DollarValue >= g.currentThreshold {
		return g.finalizeBar(), nil
	}

	return nil, nil
}

// ShouldSkipTrade returns true if the trade has already been processed.
// Use this for efficient early-skip before parsing full trade data.
func (g *Generator) ShouldSkipTrade(tradeID int64) bool {
	return tradeID > 0 && tradeID <= g.lastProcessedTradeID
}

// GetLastProcessedTradeID returns the last processed trade ID.
func (g *Generator) GetLastProcessedTradeID() int64 {
	return g.lastProcessedTradeID
}

// handleDayChange checks if the day has changed and updates threshold.
// Threshold is updated at UTC 00:00 using data from t-14 to t-1.
func (g *Generator) handleDayChange(timestamp int64) {
	dayStart := getDayStartUTC(timestamp)

	// First trade ever
	if g.currentDayStart == 0 {
		g.currentDayStart = dayStart
		return
	}

	// Same day, nothing to do
	if dayStart == g.currentDayStart {
		return
	}

	// Day changed!
	yesterdayVolume := g.currentDayVolume
	g.currentDayVolume = 0
	g.currentDayStart = dayStart

	// During warmup: collect daily volumes for SMA initialization
	if g.isWarmup {
		g.saveDailyVolume(yesterdayVolume)

		// Check if warmup is complete (we now have 14 days of data)
		if g.dailyVolumeCount >= WarmupDays {
			g.isWarmup = false
			g.initializeEMA()
			g.currentThreshold = g.calculateThreshold()
		}
		return
	}

	// After warmup: update EMA and recalculate threshold
	g.updateEMA(yesterdayVolume)
	g.currentThreshold = g.calculateThreshold()
}

// saveDailyVolume adds a daily volume to the circular buffer.
func (g *Generator) saveDailyVolume(volume float64) {
	g.dailyVolumes[g.dailyVolumeHead] = volume
	g.dailyVolumeHead = (g.dailyVolumeHead + 1) % WarmupDays

	if g.dailyVolumeCount < WarmupDays {
		g.dailyVolumeCount++
	}
}

// calculateThreshold computes threshold using EMA of daily volume.
// Formula: Threshold = EMA_14(DailyVolume) / TargetBarsPerDay
func (g *Generator) calculateThreshold() float64 {
	if g.emaVolume <= 0 {
		return 0
	}
	return g.emaVolume / float64(g.targetBarsPerDay)
}

// initializeEMA initializes EMA using SMA of collected warmup data.
// Called once when warmup period ends.
func (g *Generator) initializeEMA() {
	if g.dailyVolumeCount == 0 {
		g.emaVolume = 0
		return
	}

	var sum float64
	for i := 0; i < g.dailyVolumeCount; i++ {
		sum += g.dailyVolumes[i]
	}

	// Initialize EMA with SMA
	g.emaVolume = sum / float64(g.dailyVolumeCount)
}

// updateEMA updates the EMA with a new daily volume.
// EMA_new = alpha * newValue + (1 - alpha) * EMA_old
func (g *Generator) updateEMA(newDailyVolume float64) {
	if g.emaVolume == 0 {
		// First update after warmup or fresh start
		g.emaVolume = newDailyVolume
	} else {
		g.emaVolume = EMAAlpha*newDailyVolume + (1-EMAAlpha)*g.emaVolume
	}
}

// finalizeBar completes the current bar and resets for the next one.
func (g *Generator) finalizeBar() *DynamicDollarBar {
	bar := g.acc.Finalize(g.currentThreshold)
	g.acc.Reset()
	return &bar
}

// Flush forces the current accumulator to emit a bar (for end of data).
// Returns nil if no trades accumulated or still in warmup.
func (g *Generator) Flush() *DynamicDollarBar {
	if g.isWarmup || g.acc.TickCount == 0 {
		return nil
	}
	return g.finalizeBar()
}

// FlushDay forces saving the current day's volume without a day change.
// Call this at end of data processing to ensure the last day is counted.
func (g *Generator) FlushDay() {
	if g.currentDayVolume > 0 {
		g.saveDailyVolume(g.currentDayVolume)
		g.currentDayVolume = 0
	}
}

// GetCurrentThreshold returns the current threshold being used.
func (g *Generator) GetCurrentThreshold() float64 {
	return g.currentThreshold
}

// IsWarmup returns true if still in warmup period.
func (g *Generator) IsWarmup() bool {
	return g.isWarmup
}

// GetWarmupDaysRemaining returns how many more days of warmup are needed.
func (g *Generator) GetWarmupDaysRemaining() int {
	if !g.isWarmup {
		return 0
	}
	return WarmupDays - g.dailyVolumeCount
}

// GetPendingBar returns the current accumulator state (for inspection).
func (g *Generator) GetPendingBar() *BarAccumulator {
	if g.acc.TickCount == 0 {
		return nil
	}
	return &g.acc
}

// getDayStartUTC returns the UTC 00:00 timestamp for the given timestamp.
func getDayStartUTC(timestamp int64) int64 {
	t := time.UnixMilli(timestamp).UTC()
	dayStart := time.Date(t.Year(), t.Month(), t.Day(), 0, 0, 0, 0, time.UTC)
	return dayStart.UnixMilli()
}

// MonthCheckpoint stores the generator state at the end of a completed month.
// Used for verification and rollback on restart.
type MonthCheckpoint struct {
	Status       string `json:"status"` // "complete" or "partial"
	BarCount     int    `json:"bar_count"`
	FirstTradeID int64  `json:"first_trade_id"`
	LastTradeID  int64  `json:"last_trade_id"`
	FirstBarTime int64  `json:"first_bar_time"`
	LastBarTime  int64  `json:"last_bar_time"`

	// Generator state snapshot at end of this month (for rollback)
	EMAVolumeSnapshot   float64 `json:"ema_volume_snapshot"`
	ThresholdSnapshot   float64 `json:"threshold_snapshot"`
	DayStartSnapshot    int64   `json:"day_start_snapshot"`
	DayVolumeSnapshot   float64 `json:"day_volume_snapshot"`
	IsWarmupSnapshot    bool    `json:"is_warmup_snapshot"`
}

// GeneratorState holds the complete state for serialization.
// Critical for crash recovery - prevents double counting on resume.
type GeneratorState struct {
	// Configuration
	TargetBarsPerDay int `json:"target_bars_per_day"`

	// Daily volume tracking (used during warmup)
	DailyVolumes     [WarmupDays]float64 `json:"daily_volumes"`
	DailyVolumeHead  int                 `json:"daily_volume_head"`
	DailyVolumeCount int                 `json:"daily_volume_count"`

	// EMA tracking (used after warmup)
	EMAVolume float64 `json:"ema_volume"`

	// Current day tracking
	CurrentDayStart  int64   `json:"current_day_start"`
	CurrentDayVolume float64 `json:"current_day_volume"`

	// Threshold
	CurrentThreshold float64 `json:"current_threshold"`

	// Accumulator state
	Accumulator BarAccumulator `json:"accumulator"`

	// Warmup state
	IsWarmup bool `json:"is_warmup"`

	// Resume tracking - CRITICAL for crash recovery
	// Without this, trades before crash would be double-counted on resume
	LastProcessedTradeID int64 `json:"last_processed_trade_id"`

	// Monthly checkpoints for completeness verification and rollback
	MonthlyCheckpoints map[string]MonthCheckpoint `json:"monthly_checkpoints"`

	// Current processing month (for partial detection)
	CurrentProcessingMonth string `json:"current_processing_month"`
}

// ExportState exports the generator state for serialization.
func (g *Generator) ExportState() GeneratorState {
	// Copy checkpoints map
	checkpoints := make(map[string]MonthCheckpoint)
	for k, v := range g.monthlyCheckpoints {
		checkpoints[k] = v
	}

	return GeneratorState{
		TargetBarsPerDay:       g.targetBarsPerDay,
		DailyVolumes:           g.dailyVolumes,
		DailyVolumeHead:        g.dailyVolumeHead,
		DailyVolumeCount:       g.dailyVolumeCount,
		EMAVolume:              g.emaVolume,
		CurrentDayStart:        g.currentDayStart,
		CurrentDayVolume:       g.currentDayVolume,
		CurrentThreshold:       g.currentThreshold,
		Accumulator:            g.acc,
		IsWarmup:               g.isWarmup,
		LastProcessedTradeID:   g.lastProcessedTradeID,
		MonthlyCheckpoints:     checkpoints,
		CurrentProcessingMonth: g.currentProcessingMonth,
	}
}

// ImportState restores the generator state from a saved state.
func (g *Generator) ImportState(state GeneratorState) {
	// Restore configuration (use default if not set in state for backward compatibility)
	if state.TargetBarsPerDay > 0 {
		g.targetBarsPerDay = state.TargetBarsPerDay
	} else {
		g.targetBarsPerDay = DefaultTargetBarsPerDay
	}

	g.dailyVolumes = state.DailyVolumes
	g.dailyVolumeHead = state.DailyVolumeHead
	g.dailyVolumeCount = state.DailyVolumeCount
	g.emaVolume = state.EMAVolume
	g.currentDayStart = state.CurrentDayStart
	g.currentDayVolume = state.CurrentDayVolume
	g.currentThreshold = state.CurrentThreshold
	g.acc = state.Accumulator
	g.isWarmup = state.IsWarmup
	g.lastProcessedTradeID = state.LastProcessedTradeID
	g.currentProcessingMonth = state.CurrentProcessingMonth

	// Copy checkpoints map
	if state.MonthlyCheckpoints != nil {
		g.monthlyCheckpoints = make(map[string]MonthCheckpoint)
		for k, v := range state.MonthlyCheckpoints {
			g.monthlyCheckpoints[k] = v
		}
	}
}

// SetCurrentMonth sets the current processing month.
func (g *Generator) SetCurrentMonth(month string) {
	g.currentProcessingMonth = month
}

// GetCurrentMonth returns the current processing month.
func (g *Generator) GetCurrentMonth() string {
	return g.currentProcessingMonth
}

// IsMonthComplete checks if a month is marked as complete.
func (g *Generator) IsMonthComplete(month string) bool {
	if cp, exists := g.monthlyCheckpoints[month]; exists {
		return cp.Status == "complete"
	}
	return false
}

// GetMonthCheckpoint returns the checkpoint for a month.
func (g *Generator) GetMonthCheckpoint(month string) (MonthCheckpoint, bool) {
	cp, exists := g.monthlyCheckpoints[month]
	return cp, exists
}

// MarkMonthComplete marks a month as complete with the given metadata.
func (g *Generator) MarkMonthComplete(month string, barCount int, firstTradeID, lastTradeID, firstBarTime, lastBarTime int64) {
	g.monthlyCheckpoints[month] = MonthCheckpoint{
		Status:            "complete",
		BarCount:          barCount,
		FirstTradeID:      firstTradeID,
		LastTradeID:       lastTradeID,
		FirstBarTime:      firstBarTime,
		LastBarTime:       lastBarTime,
		EMAVolumeSnapshot: g.emaVolume,
		ThresholdSnapshot: g.currentThreshold,
		DayStartSnapshot:  g.currentDayStart,
		DayVolumeSnapshot: g.currentDayVolume,
		IsWarmupSnapshot:  g.isWarmup,
	}
}

// MarkMonthPartial marks a month as partially processed.
func (g *Generator) MarkMonthPartial(month string) {
	g.monthlyCheckpoints[month] = MonthCheckpoint{
		Status: "partial",
	}
}

// RestoreFromCheckpoint restores generator state from a month checkpoint.
// Used when needing to reprocess from a specific point.
func (g *Generator) RestoreFromCheckpoint(cp MonthCheckpoint) {
	g.emaVolume = cp.EMAVolumeSnapshot
	g.currentThreshold = cp.ThresholdSnapshot
	g.currentDayStart = cp.DayStartSnapshot
	g.currentDayVolume = cp.DayVolumeSnapshot
	g.isWarmup = cp.IsWarmupSnapshot
	g.lastProcessedTradeID = cp.LastTradeID
	g.acc = BarAccumulator{} // Reset accumulator
}

// GetPreviousMonthCheckpoint returns the checkpoint of the month before the given month.
func (g *Generator) GetPreviousMonthCheckpoint(month string) (MonthCheckpoint, bool) {
	// Parse month (format: "2024-01")
	var year, mon int
	fmt.Sscanf(month, "%d-%d", &year, &mon)

	// Calculate previous month
	mon--
	if mon < 1 {
		mon = 12
		year--
	}
	prevMonth := fmt.Sprintf("%d-%02d", year, mon)

	return g.GetMonthCheckpoint(prevMonth)
}

// DeleteMonthCheckpoint removes a month's checkpoint (for reprocessing).
func (g *Generator) DeleteMonthCheckpoint(month string) {
	delete(g.monthlyCheckpoints, month)
}

// GetEMAVolume returns the current EMA of daily volume.
func (g *Generator) GetEMAVolume() float64 {
	return g.emaVolume
}

// PrepareForMonth prepares the generator state for processing a specific month.
// This implements the "Stateless Start" principle:
// - Always restore from previous month's checkpoint before processing
// - Ensures no state corruption from previous runs
// Returns true if state was restored from checkpoint, false if starting fresh.
func (g *Generator) PrepareForMonth(month string) (restored bool, prevMonth string) {
	prevCp, exists := g.GetPreviousMonthCheckpoint(month)
	if !exists {
		// No previous checkpoint - this is likely the first month or warmup period
		// Reset critical state to prevent corruption
		if g.lastProcessedTradeID > 0 && !g.isWarmup {
			// We have state but no checkpoint for previous month
			// This could indicate corruption - reset to safe state
			g.lastProcessedTradeID = 0
			g.acc = BarAccumulator{}
		}
		return false, ""
	}

	// Calculate previous month string
	var year, mon int
	fmt.Sscanf(month, "%d-%d", &year, &mon)
	mon--
	if mon < 1 {
		mon = 12
		year--
	}
	prevMonth = fmt.Sprintf("%d-%02d", year, mon)

	// Restore from previous month's clean state
	g.RestoreFromCheckpoint(prevCp)
	return true, prevMonth
}

// ValidateStateForMonth checks if current state is valid for processing the given month.
// Returns an error if state appears corrupted.
func (g *Generator) ValidateStateForMonth(month string, firstTradeTimestamp int64) error {
	if g.currentDayStart == 0 {
		return nil // Fresh state, no validation needed
	}

	// Parse month to get expected time range
	var year, mon int
	fmt.Sscanf(month, "%d-%d", &year, &mon)
	monthStart := time.Date(year, time.Month(mon), 1, 0, 0, 0, 0, time.UTC).UnixMilli()
	monthEnd := time.Date(year, time.Month(mon)+1, 1, 0, 0, 0, 0, time.UTC).UnixMilli()

	// State's current day should be before or within the month we're processing
	if g.currentDayStart > monthEnd {
		gapDays := int((g.currentDayStart - monthEnd) / MillisPerDay)
		return &ErrStateCorruption{
			Message:      "state is from future relative to month being processed",
			StateDay:     g.currentDayStart,
			TradeDay:     monthStart,
			GapDays:      gapDays,
			LastTradeID:  g.lastProcessedTradeID,
			CurrentTradeID: 0,
		}
	}

	return nil
}
