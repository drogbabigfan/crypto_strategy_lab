package backtest

// Executor handles trade execution with Next Bar Entry/Exit logic.
//
// Key Rule: Signal at bar t → Entry/Exit at bar t+1 Open
// This prevents look-ahead bias for both entry and exit.
type Executor struct {
	config    Config
	costModel *CostModel

	// Current state
	position      *Position // nil if flat
	pendingSignal *Signal   // Signal waiting for entry execution
	pendingSize   float64   // Position size for pending signal
	pendingBar    int       // Bar index when signal was generated
	prevSignal    Signal    // Previous bar's signal (for exit decisions)
	prevSize      float64   // Previous bar's size (for reversal)

	// Rolling average volume for cost calculation
	volumeWindow []float64
	volumeSum    float64
	volumeIdx    int
	volumeCount  int
}

const volumeWindowSize = 100

// NewExecutor creates a new trade executor.
func NewExecutor(cfg Config) *Executor {
	return &Executor{
		config:       cfg,
		costModel:    NewCostModel(cfg),
		volumeWindow: make([]float64, volumeWindowSize),
	}
}

// Reset clears the executor state.
func (e *Executor) Reset() {
	e.position = nil
	e.pendingSignal = nil
	e.pendingSize = 1.0
	e.pendingBar = 0
	e.prevSignal = SignalNeutral
	e.prevSize = 1.0
	e.volumeWindow = make([]float64, volumeWindowSize)
	e.volumeSum = 0
	e.volumeIdx = 0
	e.volumeCount = 0
}

// updateAverageVolume updates the rolling average volume.
func (e *Executor) updateAverageVolume(volume float64) {
	// Subtract old value
	e.volumeSum -= e.volumeWindow[e.volumeIdx]
	// Add new value
	e.volumeWindow[e.volumeIdx] = volume
	e.volumeSum += volume
	// Move index
	e.volumeIdx = (e.volumeIdx + 1) % volumeWindowSize
	if e.volumeCount < volumeWindowSize {
		e.volumeCount++
	}
}

// getAverageVolume returns the rolling average volume.
func (e *Executor) getAverageVolume() float64 {
	if e.volumeCount == 0 {
		return 1.0 // Avoid division by zero
	}
	return e.volumeSum / float64(e.volumeCount)
}

// ProcessBar processes a single bar and returns a completed trade if any.
//
// The execution follows this order:
//  1. Update volume window
//  2. Check exit conditions using PREVIOUS bar's signal (next-bar exit)
//  3. Execute pending signal (entry at this bar's Open)
//  4. For reversal: if just exited and prevSignal is opposite, enter immediately
//  5. Register new signal for next bar execution
//  6. Store current signal for next bar's exit decision
//
// stopPrice: per-bar stop price from Python (used in custom_stop mode).
// For Long: if bar.Low <= stopPrice, exit at stopPrice.
// For Short: if bar.High >= stopPrice, exit at stopPrice.
func (e *Executor) ProcessBar(barIdx int, bar Bar, signal Signal, size float64, stopPrice float64) *Trade {
	// 1. Update volume window
	e.updateAverageVolume(bar.Volume)

	avgVolume := e.getAverageVolume()
	var completedTrade *Trade

	// 2. Check exit conditions
	if e.position != nil {
		switch e.config.ExitMode {
		case ExitModeSignal:
			// Signal mode: exit on opposite/neutral signal from PREVIOUS bar
			completedTrade = e.checkSignalExit(barIdx, bar, e.prevSignal, avgVolume, 0)
		case ExitModeCustomStop:
			// Custom stop mode: use per-bar stop price from Python
			// Stop is checked immediately on current bar's high/low
			completedTrade = e.checkCustomStopExit(barIdx, bar, stopPrice, avgVolume)
			// If no stop hit, also check for signal-based exit (Python provides exit price)
			if completedTrade == nil {
				completedTrade = e.checkSignalExit(barIdx, bar, e.prevSignal, avgVolume, stopPrice)
			}
		default:
			// TBM mode: exit on TP/SL/Timeout (uses bar data, not signal)
			completedTrade = e.checkExit(barIdx, bar, avgVolume)
		}
	}

	// 3. Execute pending signal (Next Bar Entry)
	if e.pendingSignal != nil && e.position == nil {
		e.executeEntry(barIdx, bar, avgVolume)
		e.pendingSignal = nil
	}

	// 4. For reversal: if we just exited and prevSignal is an entry signal, enter immediately
	// This allows exit and entry to happen at the same bar's Open (reversal)
	// NOTE: Reversal is disabled in custom_stop mode because:
	// - Python controls position via signals (0 = exit, no new entry)
	// - Opposite signal in custom_stop mode means "exit only", not "reverse"
	if completedTrade != nil && e.position == nil && e.prevSignal != SignalNeutral {
		// Skip reversal in custom_stop mode
		if e.config.ExitMode != ExitModeCustomStop {
			// We just exited and prevSignal indicates a new position
			sig := e.prevSignal
			e.pendingSignal = nil // Clear any pending
			// Execute entry immediately at same bar's Open (use prev size for reversal)
			e.executeEntryWithSignal(barIdx, bar, avgVolume, sig, e.prevSize)
		}
	}

	// 5. Register new signal for next bar execution (only if flat and no reversal happened)
	if e.position == nil && signal != SignalNeutral {
		sig := signal
		e.pendingSignal = &sig
		e.pendingSize = size
		e.pendingBar = barIdx
	}

	// 6. Store current signal and size for next bar's exit decision
	e.prevSignal = signal
	e.prevSize = size

	return completedTrade
}

// checkSignalExit checks if position should be closed due to signal change.
// In signal mode: Long closes on Short signal, Short closes on Long signal.
// Neutral signal does NOT trigger exit - it means "hold current position".
// In custom_stop mode: also exit on neutral signal (matches Python behavior).
// exitPriceHint: if > 0, use this as exit price instead of bar.Open (for Python-calculated exits).
func (e *Executor) checkSignalExit(barIdx int, bar Bar, signal Signal, avgVolume float64, exitPriceHint float64) *Trade {
	pos := e.position
	if pos == nil {
		return nil
	}

	shouldExit := false

	// Check for opposite signal (all modes)
	if pos.Direction == DirectionLong && signal == SignalShort {
		shouldExit = true
	} else if pos.Direction == DirectionShort && signal == SignalLong {
		shouldExit = true
	}

	// In custom_stop mode, also exit on neutral signal
	// This matches Python backtester behavior where signal=0 means "exit position"
	if e.config.ExitMode == ExitModeCustomStop && signal == SignalNeutral {
		shouldExit = true
	}

	if !shouldExit {
		return nil
	}

	// Exit price: use hint if provided, otherwise bar.Open
	exitPrice := bar.Open
	if exitPriceHint > 0 && exitPriceHint < 1e17 { // Valid exit price from Python
		exitPrice = exitPriceHint
	}
	holdingBars := barIdx - pos.EntryBar

	// Apply exit slippage
	actualExitPrice := e.costModel.ApplyExitSlippage(
		exitPrice,
		pos.Direction,
		bar.RealizedVol,
		e.config.RiskPerTrade,
		avgVolume,
	)

	// Calculate P&L
	grossPnL, netPnL := e.costModel.CalculatePnL(pos.EntryPrice, actualExitPrice, pos.Direction)

	trade := &Trade{
		EntryBar:    pos.EntryBar,
		ExitBar:     barIdx,
		Direction:   pos.Direction,
		EntryPrice:  pos.EntryPrice,
		ExitPrice:   actualExitPrice,
		GrossPnL:    grossPnL,
		PnL:         netPnL,
		TotalCost:   grossPnL - netPnL,
		ExitReason:  ExitReasonSignal,
		HoldingBars: holdingBars,
		Size:        pos.Size,
	}

	// Close position
	e.position = nil

	return trade
}

// executeEntry opens a new position at the bar's Open price using pending signal.
func (e *Executor) executeEntry(barIdx int, bar Bar, avgVolume float64) {
	signal := *e.pendingSignal
	e.executeEntryWithSignal(barIdx, bar, avgVolume, signal, e.pendingSize)
}

// executeEntryWithSignal opens a new position at the bar's Open price with given signal and size.
func (e *Executor) executeEntryWithSignal(barIdx int, bar Bar, avgVolume float64, signal Signal, size float64) {
	direction := Direction(signal) // -1, 0, or 1

	if direction == DirectionFlat {
		return
	}

	// Use RealizedVol for slippage, default to small value if not available
	volatility := bar.RealizedVol
	if volatility <= 0 {
		volatility = 0.001 // Default 0.1% volatility for slippage calc
	}

	// Apply entry slippage to Open price
	entryPrice := e.costModel.ApplyEntrySlippage(
		bar.Open,
		direction,
		volatility,
		e.config.RiskPerTrade,
		avgVolume,
	)

	var takeProfit, stopLoss float64
	var maxHoldBars int

	if e.config.ExitMode == ExitModeTBM {
		// TBM mode: Calculate barrier levels using volatility
		if direction == DirectionLong {
			takeProfit = bar.Open * (1 + e.config.PTMult*bar.RealizedVol)
			stopLoss = bar.Open * (1 - e.config.SLMult*bar.RealizedVol)
		} else {
			takeProfit = bar.Open * (1 - e.config.PTMult*bar.RealizedVol)
			stopLoss = bar.Open * (1 + e.config.SLMult*bar.RealizedVol)
		}
		maxHoldBars = e.config.MaxHoldBars
	}
	// Signal mode: TP/SL/MaxHold remain 0 (unused)

	e.position = &Position{
		Direction:   direction,
		EntryPrice:  entryPrice,
		EntryBar:    barIdx,
		TakeProfit:  takeProfit,
		StopLoss:    stopLoss,
		MaxHoldBars: maxHoldBars,
		Size:        size,
	}
}

// checkExit checks if the current position should be closed.
func (e *Executor) checkExit(barIdx int, bar Bar, avgVolume float64) *Trade {
	pos := e.position

	// Check holding period
	holdingBars := barIdx - pos.EntryBar

	// Determine exit condition and price
	exitPrice, exitReason := e.determineExit(bar, pos, holdingBars)

	if exitReason == "" {
		return nil // No exit
	}

	// Apply exit slippage
	actualExitPrice := e.costModel.ApplyExitSlippage(
		exitPrice,
		pos.Direction,
		bar.RealizedVol,
		e.config.RiskPerTrade,
		avgVolume,
	)

	// Calculate P&L
	grossPnL, netPnL := e.costModel.CalculatePnL(pos.EntryPrice, actualExitPrice, pos.Direction)

	trade := &Trade{
		EntryBar:    pos.EntryBar,
		ExitBar:     barIdx,
		Direction:   pos.Direction,
		EntryPrice:  pos.EntryPrice,
		ExitPrice:   actualExitPrice,
		GrossPnL:    grossPnL,
		PnL:         netPnL,
		TotalCost:   grossPnL - netPnL,
		ExitReason:  exitReason,
		HoldingBars: holdingBars,
		Size:        pos.Size,
	}

	// Close position
	e.position = nil

	return trade
}

// determineExit determines if and how a position should be exited.
// Returns the exit price and reason, or empty reason if no exit.
//
// Uses HighTime/LowTime for precise ordering when both SL and TP hit in same bar.
// Priority: First-to-occur (by timestamp) > Timeout
func (e *Executor) determineExit(bar Bar, pos *Position, holdingBars int) (float64, ExitReason) {
	if pos.Direction == DirectionLong {
		return e.determineLongExit(bar, pos, holdingBars)
	}
	return e.determineShortExit(bar, pos, holdingBars)
}

// determineLongExit checks exit conditions for a long position.
// Uses HighTime/LowTime for precise SL/TP ordering.
func (e *Executor) determineLongExit(bar Bar, pos *Position, holdingBars int) (float64, ExitReason) {
	slHit := bar.Low <= pos.StopLoss
	tpHit := bar.High >= pos.TakeProfit

	if slHit && tpHit {
		// Both hit in same bar - use actual timestamps to determine which came first
		if bar.LowTime <= bar.HighTime {
			return pos.StopLoss, ExitReasonSL
		}
		return pos.TakeProfit, ExitReasonTP
	}

	if slHit {
		return pos.StopLoss, ExitReasonSL
	}

	if tpHit {
		return pos.TakeProfit, ExitReasonTP
	}

	// Check timeout
	if holdingBars >= pos.MaxHoldBars {
		return bar.Close, ExitReasonTimeout
	}

	return 0, ""
}

// determineShortExit checks exit conditions for a short position.
// Uses HighTime/LowTime for precise SL/TP ordering.
func (e *Executor) determineShortExit(bar Bar, pos *Position, holdingBars int) (float64, ExitReason) {
	slHit := bar.High >= pos.StopLoss   // Price went up = bad for short
	tpHit := bar.Low <= pos.TakeProfit  // Price went down = good for short

	if slHit && tpHit {
		// Both hit in same bar - use actual timestamps to determine which came first
		if bar.HighTime <= bar.LowTime {
			return pos.StopLoss, ExitReasonSL
		}
		return pos.TakeProfit, ExitReasonTP
	}

	if slHit {
		return pos.StopLoss, ExitReasonSL
	}

	if tpHit {
		return pos.TakeProfit, ExitReasonTP
	}

	// Check timeout
	if holdingBars >= pos.MaxHoldBars {
		return bar.Close, ExitReasonTimeout
	}

	return 0, ""
}

// HasPosition returns true if there's an open position.
func (e *Executor) HasPosition() bool {
	return e.position != nil
}

// GetPosition returns the current position (may be nil).
func (e *Executor) GetPosition() *Position {
	return e.position
}

// HasPendingSignal returns true if there's a signal waiting for execution.
func (e *Executor) HasPendingSignal() bool {
	return e.pendingSignal != nil
}

// checkCustomStopExit checks if position should be closed due to custom stop price.
// Uses bar's high/low to determine if stop was hit, exits at stop price (not next bar open).
// This enables precise stop execution for trailing stops, MAE stops, etc.
func (e *Executor) checkCustomStopExit(barIdx int, bar Bar, stopPrice float64, avgVolume float64) *Trade {
	pos := e.position
	if pos == nil || stopPrice <= 0 {
		return nil
	}

	var slHit bool
	var exitPrice float64

	if pos.Direction == DirectionLong {
		// Long position: stop hit if low reaches stop price
		slHit = bar.Low <= stopPrice
		exitPrice = stopPrice
	} else {
		// Short position: stop hit if high reaches stop price
		slHit = bar.High >= stopPrice
		exitPrice = stopPrice
	}

	if !slHit {
		return nil
	}

	holdingBars := barIdx - pos.EntryBar

	// Apply exit slippage to stop price
	actualExitPrice := e.costModel.ApplyExitSlippage(
		exitPrice,
		pos.Direction,
		bar.RealizedVol,
		e.config.RiskPerTrade,
		avgVolume,
	)

	// Calculate P&L
	grossPnL, netPnL := e.costModel.CalculatePnL(pos.EntryPrice, actualExitPrice, pos.Direction)

	trade := &Trade{
		EntryBar:    pos.EntryBar,
		ExitBar:     barIdx,
		Direction:   pos.Direction,
		EntryPrice:  pos.EntryPrice,
		ExitPrice:   actualExitPrice,
		GrossPnL:    grossPnL,
		PnL:         netPnL,
		TotalCost:   grossPnL - netPnL,
		ExitReason:  ExitReasonSL, // Custom stop is treated as SL
		HoldingBars: holdingBars,
		Size:        pos.Size,
	}

	// Close position
	e.position = nil

	return trade
}

// ForceClose closes the current position at the given price.
// Used for end-of-backtest cleanup.
func (e *Executor) ForceClose(barIdx int, closePrice float64, avgVolume float64, volatility float64) *Trade {
	if e.position == nil {
		return nil
	}

	pos := e.position

	// Apply exit slippage
	actualExitPrice := e.costModel.ApplyExitSlippage(
		closePrice,
		pos.Direction,
		volatility,
		e.config.RiskPerTrade,
		avgVolume,
	)

	grossPnL, netPnL := e.costModel.CalculatePnL(pos.EntryPrice, actualExitPrice, pos.Direction)
	holdingBars := barIdx - pos.EntryBar

	trade := &Trade{
		EntryBar:    pos.EntryBar,
		ExitBar:     barIdx,
		Direction:   pos.Direction,
		EntryPrice:  pos.EntryPrice,
		ExitPrice:   actualExitPrice,
		GrossPnL:    grossPnL,
		PnL:         netPnL,
		TotalCost:   grossPnL - netPnL,
		ExitReason:  ExitReasonTimeout,
		HoldingBars: holdingBars,
		Size:        pos.Size,
	}

	e.position = nil
	return trade
}
