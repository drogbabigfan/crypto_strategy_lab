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
func (e *Executor) ProcessBar(barIdx int, bar Bar, signal Signal, size float64) *Trade {
	// 1. Update volume window
	e.updateAverageVolume(bar.Volume)

	avgVolume := e.getAverageVolume()
	var completedTrade *Trade

	// 2. Check exit conditions using PREVIOUS bar's signal (next-bar exit)
	// This prevents look-ahead bias: signal[i-1] decides exit at bar[i].Open
	if e.position != nil {
		if e.config.ExitMode == ExitModeSignal {
			// Signal mode: exit on opposite/neutral signal from PREVIOUS bar
			completedTrade = e.checkSignalExit(barIdx, bar, e.prevSignal, avgVolume)
		} else {
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
	if completedTrade != nil && e.position == nil && e.prevSignal != SignalNeutral {
		// We just exited and prevSignal indicates a new position
		sig := e.prevSignal
		e.pendingSignal = nil // Clear any pending
		// Execute entry immediately at same bar's Open (use prev size for reversal)
		e.executeEntryWithSignal(barIdx, bar, avgVolume, sig, e.prevSize)
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

// checkSignalExit checks if position should be closed due to opposite or neutral signal.
// In signal mode: Long closes on Short/Neutral signal, Short closes on Long/Neutral signal.
func (e *Executor) checkSignalExit(barIdx int, bar Bar, signal Signal, avgVolume float64) *Trade {
	pos := e.position
	if pos == nil {
		return nil
	}

	// Check for opposite or neutral signal
	shouldExit := false
	if pos.Direction == DirectionLong && (signal == SignalShort || signal == SignalNeutral) {
		shouldExit = true
	} else if pos.Direction == DirectionShort && (signal == SignalLong || signal == SignalNeutral) {
		shouldExit = true
	}

	if !shouldExit {
		return nil
	}

	// Exit at this bar's Open (signal was generated at previous bar)
	exitPrice := bar.Open
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
