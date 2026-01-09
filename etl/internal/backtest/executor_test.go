package backtest

import (
	"testing"
)

func TestNewExecutor(t *testing.T) {
	cfg := DefaultConfig()
	e := NewExecutor(cfg)

	if e.position != nil {
		t.Error("New executor should have no position")
	}
	if e.pendingSignal != nil {
		t.Error("New executor should have no pending signal")
	}
	if e.HasPosition() {
		t.Error("HasPosition should return false")
	}
	if e.HasPendingSignal() {
		t.Error("HasPendingSignal should return false")
	}
}

func TestNextBarEntry(t *testing.T) {
	// Core test: Signal at bar 0 → Entry at bar 1 Open
	cfg := DefaultConfig()
	cfg.SLMult = 2.0
	cfg.PTMult = 2.0
	cfg.MaxHoldBars = 100
	e := NewExecutor(cfg)

	bar0 := Bar{
		Open: 50000, High: 50100, Low: 49900, Close: 50050,
		Volume: 1000, RealizedVol: 0.02,
	}
	bar1 := Bar{
		Open: 50100, High: 50200, Low: 50000, Close: 50150,
		Volume: 1000, RealizedVol: 0.02,
	}

	// Bar 0: Generate long signal, no entry yet
	trade := e.ProcessBar(0, bar0, SignalLong, 1.0, 0.0)
	if trade != nil {
		t.Error("Should not complete trade on signal bar")
	}
	if e.HasPosition() {
		t.Error("Should not have position on signal bar")
	}
	if !e.HasPendingSignal() {
		t.Error("Should have pending signal after bar 0")
	}

	// Bar 1: Entry should happen at Open
	trade = e.ProcessBar(1, bar1, SignalNeutral, 1.0, 0.0)
	if trade != nil {
		t.Error("Should not complete trade on entry bar")
	}
	if !e.HasPosition() {
		t.Fatal("Should have position after bar 1")
	}

	pos := e.GetPosition()
	if pos.EntryBar != 1 {
		t.Errorf("Entry bar should be 1, got %d", pos.EntryBar)
	}
	if pos.Direction != DirectionLong {
		t.Errorf("Direction should be Long, got %d", pos.Direction)
	}
	// Entry price should be bar1.Open with slippage applied
	if pos.EntryPrice <= bar1.Open {
		t.Error("Long entry should have slippage applied (higher than Open)")
	}
}

func TestLongTakeProfitExit(t *testing.T) {
	cfg := DefaultConfig()
	cfg.SLMult = 2.0
	cfg.PTMult = 2.0
	cfg.MaxHoldBars = 100
	e := NewExecutor(cfg)

	// Setup: vol = 0.02, so TP = Open * (1 + 0.04) = 1.04 * Open
	bar0 := Bar{
		Open: 50000, High: 50100, Low: 49900, Close: 50050,
		Volume: 1000, RealizedVol: 0.02,
	}

	// Bar 0: Signal
	e.ProcessBar(0, bar0, SignalLong, 1.0, 0.0)

	// Bar 1: Entry at Open = 50000, TP = 52000
	bar1 := Bar{
		Open: 50000, High: 50100, Low: 49950, Close: 50050,
		Volume: 1000, RealizedVol: 0.02,
	}
	e.ProcessBar(1, bar1, SignalNeutral, 1.0, 0.0)

	pos := e.GetPosition()
	t.Logf("Entry: %f, TP: %f, SL: %f", pos.EntryPrice, pos.TakeProfit, pos.StopLoss)

	// Bar 2: Price hits TP (High >= 52000)
	bar2 := Bar{
		Open: 50050, High: 52100, Low: 50000, Close: 51800,
		Volume: 1000, RealizedVol: 0.02,
	}
	trade := e.ProcessBar(2, bar2, SignalNeutral, 1.0, 0.0)

	if trade == nil {
		t.Fatal("Should have closed trade on TP hit")
	}
	if trade.ExitReason != ExitReasonTP {
		t.Errorf("Exit reason should be TP, got %s", trade.ExitReason)
	}
	if trade.PnL <= 0 {
		t.Errorf("Long TP exit should be profitable, got PnL %f", trade.PnL)
	}
	if e.HasPosition() {
		t.Error("Should not have position after exit")
	}
}

func TestLongStopLossExit(t *testing.T) {
	cfg := DefaultConfig()
	cfg.SLMult = 2.0
	cfg.PTMult = 2.0
	cfg.MaxHoldBars = 100
	e := NewExecutor(cfg)

	bar0 := Bar{
		Open: 50000, High: 50100, Low: 49900, Close: 50050,
		Volume: 1000, RealizedVol: 0.02,
	}
	e.ProcessBar(0, bar0, SignalLong, 1.0, 0.0)

	// Bar 1: Entry, SL = 50000 * (1 - 0.04) = 48000
	bar1 := Bar{
		Open: 50000, High: 50100, Low: 49950, Close: 50050,
		Volume: 1000, RealizedVol: 0.02,
	}
	e.ProcessBar(1, bar1, SignalNeutral, 1.0, 0.0)

	// Bar 2: Price hits SL
	bar2 := Bar{
		Open: 49950, High: 50000, Low: 47800, Close: 48100,
		Volume: 1000, RealizedVol: 0.02,
	}
	trade := e.ProcessBar(2, bar2, SignalNeutral, 1.0, 0.0)

	if trade == nil {
		t.Fatal("Should have closed trade on SL hit")
	}
	if trade.ExitReason != ExitReasonSL {
		t.Errorf("Exit reason should be SL, got %s", trade.ExitReason)
	}
	if trade.PnL >= 0 {
		t.Errorf("Long SL exit should be losing, got PnL %f", trade.PnL)
	}
}

func TestShortTakeProfitExit(t *testing.T) {
	cfg := DefaultConfig()
	cfg.SLMult = 2.0
	cfg.PTMult = 2.0
	cfg.MaxHoldBars = 100
	e := NewExecutor(cfg)

	bar0 := Bar{
		Open: 50000, High: 50100, Low: 49900, Close: 49950,
		Volume: 1000, RealizedVol: 0.02,
	}
	e.ProcessBar(0, bar0, SignalShort, 1.0, 0.0)

	// Bar 1: Entry, TP = 50000 * (1 - 0.04) = 48000
	bar1 := Bar{
		Open: 50000, High: 50100, Low: 49950, Close: 49980,
		Volume: 1000, RealizedVol: 0.02,
	}
	e.ProcessBar(1, bar1, SignalNeutral, 1.0, 0.0)

	pos := e.GetPosition()
	if pos.Direction != DirectionShort {
		t.Fatalf("Expected short position, got %d", pos.Direction)
	}
	t.Logf("Short Entry: %f, TP: %f, SL: %f", pos.EntryPrice, pos.TakeProfit, pos.StopLoss)

	// Bar 2: Price hits TP (Low <= 48000)
	bar2 := Bar{
		Open: 49900, High: 50000, Low: 47800, Close: 48100,
		Volume: 1000, RealizedVol: 0.02,
	}
	trade := e.ProcessBar(2, bar2, SignalNeutral, 1.0, 0.0)

	if trade == nil {
		t.Fatal("Should have closed short on TP hit")
	}
	if trade.ExitReason != ExitReasonTP {
		t.Errorf("Exit reason should be TP, got %s", trade.ExitReason)
	}
	if trade.PnL <= 0 {
		t.Errorf("Short TP exit should be profitable, got PnL %f", trade.PnL)
	}
}

func TestShortStopLossExit(t *testing.T) {
	cfg := DefaultConfig()
	cfg.SLMult = 2.0
	cfg.PTMult = 2.0
	cfg.MaxHoldBars = 100
	e := NewExecutor(cfg)

	bar0 := Bar{
		Open: 50000, High: 50100, Low: 49900, Close: 49950,
		Volume: 1000, RealizedVol: 0.02,
	}
	e.ProcessBar(0, bar0, SignalShort, 1.0, 0.0)

	// Bar 1: Entry, SL = 50000 * (1 + 0.04) = 52000
	bar1 := Bar{
		Open: 50000, High: 50100, Low: 49950, Close: 50050,
		Volume: 1000, RealizedVol: 0.02,
	}
	e.ProcessBar(1, bar1, SignalNeutral, 1.0, 0.0)

	// Bar 2: Price hits SL (High >= 52000)
	bar2 := Bar{
		Open: 50100, High: 52200, Low: 50000, Close: 51900,
		Volume: 1000, RealizedVol: 0.02,
	}
	trade := e.ProcessBar(2, bar2, SignalNeutral, 1.0, 0.0)

	if trade == nil {
		t.Fatal("Should have closed short on SL hit")
	}
	if trade.ExitReason != ExitReasonSL {
		t.Errorf("Exit reason should be SL, got %s", trade.ExitReason)
	}
	if trade.PnL >= 0 {
		t.Errorf("Short SL exit should be losing, got PnL %f", trade.PnL)
	}
}

func TestTimeoutExit(t *testing.T) {
	cfg := DefaultConfig()
	cfg.SLMult = 2.0
	cfg.PTMult = 2.0
	cfg.MaxHoldBars = 3 // Short hold period for testing
	e := NewExecutor(cfg)

	bar0 := Bar{
		Open: 50000, High: 50100, Low: 49900, Close: 50050,
		Volume: 1000, RealizedVol: 0.02,
	}
	e.ProcessBar(0, bar0, SignalLong, 1.0, 0.0)

	// Bar 1: Entry
	e.ProcessBar(1, bar0, SignalNeutral, 1.0, 0.0)

	// Bars 2-3: No exit condition hit
	for i := 2; i < 4; i++ {
		trade := e.ProcessBar(i, bar0, SignalNeutral, 1.0, 0.0)
		if i < 4 && trade != nil {
			t.Errorf("Should not exit before MaxHoldBars, exited at bar %d", i)
		}
	}

	// Bar 4: Timeout (held for 3 bars: 1, 2, 3)
	trade := e.ProcessBar(4, bar0, SignalNeutral, 1.0, 0.0)
	if trade == nil {
		t.Fatal("Should have exited on timeout")
	}
	if trade.ExitReason != ExitReasonTimeout {
		t.Errorf("Exit reason should be TIMEOUT, got %s", trade.ExitReason)
	}
	if trade.HoldingBars != 3 {
		t.Errorf("Holding bars should be 3, got %d", trade.HoldingBars)
	}
}

func TestNoEntryOnNeutralSignal(t *testing.T) {
	cfg := DefaultConfig()
	e := NewExecutor(cfg)

	bar := Bar{
		Open: 50000, High: 50100, Low: 49900, Close: 50050,
		Volume: 1000, RealizedVol: 0.02,
	}

	// Multiple neutral signals should not create positions
	for i := 0; i < 5; i++ {
		trade := e.ProcessBar(i, bar, SignalNeutral, 1.0, 0.0)
		if trade != nil {
			t.Error("Neutral signal should not produce trade")
		}
		if e.HasPosition() {
			t.Error("Neutral signal should not create position")
		}
	}
}

func TestSLHitBeforeTP(t *testing.T) {
	// When both SL and TP hit in same bar, use timestamps to determine order
	cfg := DefaultConfig()
	cfg.SLMult = 2.0
	cfg.PTMult = 2.0
	cfg.MaxHoldBars = 100

	t.Run("SL first by time for long", func(t *testing.T) {
		e := NewExecutor(cfg)

		bar0 := Bar{
			Open: 50000, High: 50100, Low: 49900, Close: 50050,
			Volume: 1000, RealizedVol: 0.02,
		}
		e.ProcessBar(0, bar0, SignalLong, 1.0, 0.0)

		bar1 := Bar{
			Open: 50000, High: 50100, Low: 49900, Close: 50050,
			Volume: 1000, RealizedVol: 0.02,
		}
		e.ProcessBar(1, bar1, SignalNeutral, 1.0, 0.0)

		// Bar hits both SL and TP, but Low (SL trigger) happened first
		bar2 := Bar{
			Open: 50000, High: 53000, Low: 47000, Close: 50000,
			HighTime: 2000, LowTime: 1000, // Low happened first
			Volume: 1000, RealizedVol: 0.02,
		}
		trade := e.ProcessBar(2, bar2, SignalNeutral, 1.0, 0.0)

		if trade == nil {
			t.Fatal("Should have exited")
		}
		if trade.ExitReason != ExitReasonSL {
			t.Errorf("SL should trigger first, got %s", trade.ExitReason)
		}
	})

	t.Run("TP first by time for long", func(t *testing.T) {
		e := NewExecutor(cfg)

		bar0 := Bar{
			Open: 50000, High: 50100, Low: 49900, Close: 50050,
			Volume: 1000, RealizedVol: 0.02,
		}
		e.ProcessBar(0, bar0, SignalLong, 1.0, 0.0)

		bar1 := Bar{
			Open: 50000, High: 50100, Low: 49900, Close: 50050,
			Volume: 1000, RealizedVol: 0.02,
		}
		e.ProcessBar(1, bar1, SignalNeutral, 1.0, 0.0)

		// Bar hits both SL and TP, but High (TP trigger) happened first
		bar2 := Bar{
			Open: 50000, High: 53000, Low: 47000, Close: 50000,
			HighTime: 1000, LowTime: 2000, // High happened first
			Volume: 1000, RealizedVol: 0.02,
		}
		trade := e.ProcessBar(2, bar2, SignalNeutral, 1.0, 0.0)

		if trade == nil {
			t.Fatal("Should have exited")
		}
		if trade.ExitReason != ExitReasonTP {
			t.Errorf("TP should trigger first, got %s", trade.ExitReason)
		}
	})

	t.Run("SL first by time for short", func(t *testing.T) {
		e := NewExecutor(cfg)

		bar0 := Bar{
			Open: 50000, High: 50100, Low: 49900, Close: 50050,
			Volume: 1000, RealizedVol: 0.02,
		}
		e.ProcessBar(0, bar0, SignalShort, 1.0, 0.0)

		bar1 := Bar{
			Open: 50000, High: 50100, Low: 49900, Close: 50050,
			Volume: 1000, RealizedVol: 0.02,
		}
		e.ProcessBar(1, bar1, SignalNeutral, 1.0, 0.0)

		// Bar hits both SL (52000) and TP (48000), High (SL trigger) happened first
		bar2 := Bar{
			Open: 50000, High: 53000, Low: 47000, Close: 50000,
			HighTime: 1000, LowTime: 2000, // High happened first = SL for short
			Volume: 1000, RealizedVol: 0.02,
		}
		trade := e.ProcessBar(2, bar2, SignalNeutral, 1.0, 0.0)

		if trade == nil {
			t.Fatal("Should have exited")
		}
		if trade.ExitReason != ExitReasonSL {
			t.Errorf("SL should trigger first for short, got %s", trade.ExitReason)
		}
	})

	t.Run("TP first by time for short", func(t *testing.T) {
		e := NewExecutor(cfg)

		bar0 := Bar{
			Open: 50000, High: 50100, Low: 49900, Close: 50050,
			Volume: 1000, RealizedVol: 0.02,
		}
		e.ProcessBar(0, bar0, SignalShort, 1.0, 0.0)

		bar1 := Bar{
			Open: 50000, High: 50100, Low: 49900, Close: 50050,
			Volume: 1000, RealizedVol: 0.02,
		}
		e.ProcessBar(1, bar1, SignalNeutral, 1.0, 0.0)

		// Bar hits both SL and TP, Low (TP trigger for short) happened first
		bar2 := Bar{
			Open: 50000, High: 53000, Low: 47000, Close: 50000,
			HighTime: 2000, LowTime: 1000, // Low happened first = TP for short
			Volume: 1000, RealizedVol: 0.02,
		}
		trade := e.ProcessBar(2, bar2, SignalNeutral, 1.0, 0.0)

		if trade == nil {
			t.Fatal("Should have exited")
		}
		if trade.ExitReason != ExitReasonTP {
			t.Errorf("TP should trigger first for short, got %s", trade.ExitReason)
		}
	})
}

func TestNoSignalWhileInPosition(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MaxHoldBars = 100
	e := NewExecutor(cfg)

	bar := Bar{
		Open: 50000, High: 50100, Low: 49900, Close: 50050,
		Volume: 1000, RealizedVol: 0.02,
	}

	// Bar 0: Long signal
	e.ProcessBar(0, bar, SignalLong, 1.0, 0.0)
	// Bar 1: Entry
	e.ProcessBar(1, bar, SignalNeutral, 1.0, 0.0)

	if !e.HasPosition() {
		t.Fatal("Should have position")
	}

	// Bar 2: New signal should not register while in position
	e.ProcessBar(2, bar, SignalShort, 1.0, 0.0)

	// Position should still be Long
	if e.GetPosition().Direction != DirectionLong {
		t.Error("Position direction should not change while in position")
	}
}

func TestForceClose(t *testing.T) {
	cfg := DefaultConfig()
	e := NewExecutor(cfg)

	bar := Bar{
		Open: 50000, High: 50100, Low: 49900, Close: 50050,
		Volume: 1000, RealizedVol: 0.02,
	}

	// Create a position
	e.ProcessBar(0, bar, SignalLong, 1.0, 0.0)
	e.ProcessBar(1, bar, SignalNeutral, 1.0, 0.0)

	if !e.HasPosition() {
		t.Fatal("Should have position for force close test")
	}

	// Force close at bar 5
	trade := e.ForceClose(5, 51000, 1000, 0.02)

	if trade == nil {
		t.Fatal("ForceClose should return a trade")
	}
	if trade.ExitReason != ExitReasonTimeout {
		t.Errorf("ForceClose exit reason should be TIMEOUT, got %s", trade.ExitReason)
	}
	if e.HasPosition() {
		t.Error("Should not have position after ForceClose")
	}
}

func TestForceCloseNoPosition(t *testing.T) {
	e := NewExecutor(DefaultConfig())

	trade := e.ForceClose(0, 50000, 1000, 0.02)
	if trade != nil {
		t.Error("ForceClose with no position should return nil")
	}
}

func TestReset(t *testing.T) {
	cfg := DefaultConfig()
	e := NewExecutor(cfg)

	bar := Bar{
		Open: 50000, High: 50100, Low: 49900, Close: 50050,
		Volume: 1000, RealizedVol: 0.02,
	}

	// Create state
	e.ProcessBar(0, bar, SignalLong, 1.0, 0.0)
	e.ProcessBar(1, bar, SignalNeutral, 1.0, 0.0)

	if !e.HasPosition() {
		t.Fatal("Should have position before reset")
	}

	e.Reset()

	if e.HasPosition() {
		t.Error("Should not have position after reset")
	}
	if e.HasPendingSignal() {
		t.Error("Should not have pending signal after reset")
	}
}

func TestVolumeWindowUpdate(t *testing.T) {
	e := NewExecutor(DefaultConfig())

	// Process bars with known volumes
	volumes := []float64{100, 200, 300, 400, 500}
	for i, vol := range volumes {
		bar := Bar{
			Open: 50000, High: 50100, Low: 49900, Close: 50050,
			Volume: vol, RealizedVol: 0.02,
		}
		e.ProcessBar(i, bar, SignalNeutral, 1.0, 0.0)
	}

	expectedAvg := (100 + 200 + 300 + 400 + 500) / 5.0
	gotAvg := e.getAverageVolume()

	if gotAvg != expectedAvg {
		t.Errorf("Average volume = %f, want %f", gotAvg, expectedAvg)
	}
}

func TestHoldingBarsCalculation(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MaxHoldBars = 10
	e := NewExecutor(cfg)

	bar := Bar{
		Open: 50000, High: 50100, Low: 49900, Close: 50050,
		Volume: 1000, RealizedVol: 0.02,
	}

	// Bar 0: Signal
	e.ProcessBar(0, bar, SignalLong, 1.0, 0.0)
	// Bar 1: Entry
	e.ProcessBar(1, bar, SignalNeutral, 1.0, 0.0)
	// Bars 2-10: Wait for timeout
	var trade *Trade
	for i := 2; i <= 11; i++ {
		trade = e.ProcessBar(i, bar, SignalNeutral, 1.0, 0.0)
	}

	if trade == nil {
		t.Fatal("Should have exited on timeout")
	}
	// Entered at bar 1, exited at bar 11, holding = 10
	if trade.HoldingBars != 10 {
		t.Errorf("HoldingBars = %d, want 10", trade.HoldingBars)
	}
}

// Benchmark tests
func BenchmarkProcessBar(b *testing.B) {
	cfg := DefaultConfig()
	e := NewExecutor(cfg)

	bar := Bar{
		Open: 50000, High: 50100, Low: 49900, Close: 50050,
		Volume: 1000, RealizedVol: 0.02,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		e.ProcessBar(i%1000, bar, Signal(i%3-1), 1.0, 0.0)
	}
}

// ============== Signal Mode Tests ==============

func TestSignalModeBasic(t *testing.T) {
	// Test signal mode: Long -> Short signal -> exit at NEXT bar (next bar exit logic)
	// Signal at bar N -> Exit at bar N+1 open
	cfg := DefaultConfig()
	cfg.ExitMode = ExitModeSignal

	e := NewExecutor(cfg)

	bar0 := Bar{Open: 100, High: 101, Low: 99, Close: 100, Volume: 1000}
	bar1 := Bar{Open: 100, High: 102, Low: 99, Close: 101, Volume: 1000}
	bar2 := Bar{Open: 101, High: 103, Low: 100, Close: 102, Volume: 1000}
	bar3 := Bar{Open: 102, High: 104, Low: 101, Close: 103, Volume: 1000} // Short signal here
	bar4 := Bar{Open: 103, High: 105, Low: 102, Close: 104, Volume: 1000} // Exit happens here

	// Bar 0: Long signal (pending)
	trade := e.ProcessBar(0, bar0, SignalLong, 1.0, 0.0)
	if trade != nil {
		t.Error("Should not have trade on signal bar")
	}
	if !e.HasPendingSignal() {
		t.Error("Should have pending signal")
	}

	// Bar 1: Entry at Open
	trade = e.ProcessBar(1, bar1, SignalNeutral, 1.0, 0.0)
	if trade != nil {
		t.Error("Should not complete trade on entry bar")
	}
	if !e.HasPosition() {
		t.Error("Should have position after entry")
	}
	pos := e.GetPosition()
	if pos.Direction != DirectionLong {
		t.Error("Should be long position")
	}

	// Bar 2: Hold (neutral signal)
	trade = e.ProcessBar(2, bar2, SignalNeutral, 1.0, 0.0)
	if trade != nil {
		t.Error("Should not exit on neutral signal")
	}
	if !e.HasPosition() {
		t.Error("Should still have position")
	}

	// Bar 3: Short signal issued (exit will happen at bar 4)
	trade = e.ProcessBar(3, bar3, SignalShort, 1.0, 0.0)
	if trade != nil {
		t.Error("Should not exit yet - next bar exit logic")
	}
	if !e.HasPosition() {
		t.Error("Should still have position until next bar")
	}

	// Bar 4: Exit happens at Open (due to Short signal from bar 3)
	// Note: Reversal logic kicks in - Short position is opened immediately
	trade = e.ProcessBar(4, bar4, SignalNeutral, 1.0, 0.0)
	if trade == nil {
		t.Fatal("Should exit on this bar due to previous Short signal")
	}
	if trade.ExitReason != ExitReasonSignal {
		t.Errorf("Exit reason should be SIGNAL, got %s", trade.ExitReason)
	}
	if trade.Direction != DirectionLong {
		t.Error("Trade direction should be Long")
	}
	// After exit, reversal logic opens a Short position (because prevSignal was Short)
	if !e.HasPosition() {
		t.Error("Should have Short position due to reversal")
	}
	pos = e.GetPosition()
	if pos.Direction != DirectionShort {
		t.Errorf("Should have Short position after reversal, got %d", pos.Direction)
	}
}

func TestSignalModeShortToLong(t *testing.T) {
	// Test signal mode: Short -> Long signal -> exit at NEXT bar
	cfg := DefaultConfig()
	cfg.ExitMode = ExitModeSignal

	e := NewExecutor(cfg)

	bar0 := Bar{Open: 100, High: 101, Low: 99, Close: 100, Volume: 1000}
	bar1 := Bar{Open: 100, High: 101, Low: 98, Close: 99, Volume: 1000}  // Entry
	bar2 := Bar{Open: 99, High: 100, Low: 97, Close: 98, Volume: 1000}   // Hold
	bar3 := Bar{Open: 98, High: 99, Low: 96, Close: 97, Volume: 1000}    // Long signal here
	bar4 := Bar{Open: 97, High: 98, Low: 95, Close: 96, Volume: 1000}    // Exit happens here

	// Bar 0: Short signal
	e.ProcessBar(0, bar0, SignalShort, 1.0, 0.0)

	// Bar 1: Entry
	e.ProcessBar(1, bar1, SignalNeutral, 1.0, 0.0)
	if !e.HasPosition() {
		t.Fatal("Should have short position")
	}
	pos := e.GetPosition()
	if pos.Direction != DirectionShort {
		t.Errorf("Should be short, got %d", pos.Direction)
	}

	// Bar 2: Hold
	trade := e.ProcessBar(2, bar2, SignalNeutral, 1.0, 0.0)
	if trade != nil {
		t.Error("Should not exit on neutral")
	}

	// Bar 3: Long signal issued (exit will happen at bar 4)
	trade = e.ProcessBar(3, bar3, SignalLong, 1.0, 0.0)
	if trade != nil {
		t.Error("Should not exit yet - next bar exit logic")
	}

	// Bar 4: Exit happens at Open
	trade = e.ProcessBar(4, bar4, SignalNeutral, 1.0, 0.0)
	if trade == nil {
		t.Fatal("Should exit on this bar due to previous Long signal")
	}
	if trade.Direction != DirectionShort {
		t.Error("Trade should be short")
	}
	if trade.ExitReason != ExitReasonSignal {
		t.Errorf("Exit reason should be SIGNAL, got %s", trade.ExitReason)
	}
}

func TestSignalModeNoExitOnSameDirection(t *testing.T) {
	// Long position should NOT exit on another Long signal
	cfg := DefaultConfig()
	cfg.ExitMode = ExitModeSignal

	e := NewExecutor(cfg)

	bar0 := Bar{Open: 100, High: 101, Low: 99, Close: 100, Volume: 1000}
	bar1 := Bar{Open: 100, High: 102, Low: 99, Close: 101, Volume: 1000}
	bar2 := Bar{Open: 101, High: 103, Low: 100, Close: 102, Volume: 1000}

	// Enter long
	e.ProcessBar(0, bar0, SignalLong, 1.0, 0.0)
	e.ProcessBar(1, bar1, SignalNeutral, 1.0, 0.0)

	if !e.HasPosition() {
		t.Fatal("Should have position")
	}

	// Another Long signal should not exit
	trade := e.ProcessBar(2, bar2, SignalLong, 1.0, 0.0)
	if trade != nil {
		t.Error("Same direction signal should not cause exit")
	}
	if !e.HasPosition() {
		t.Error("Should still have position")
	}
}

func TestSignalModeNoTPSL(t *testing.T) {
	// In signal mode, TP/SL should not be set
	cfg := DefaultConfig()
	cfg.ExitMode = ExitModeSignal

	e := NewExecutor(cfg)

	bar0 := Bar{Open: 100, High: 101, Low: 99, Close: 100, Volume: 1000, RealizedVol: 0.02}
	bar1 := Bar{Open: 100, High: 120, Low: 80, Close: 100, Volume: 1000, RealizedVol: 0.02} // Extreme range

	// Enter long
	e.ProcessBar(0, bar0, SignalLong, 1.0, 0.0)
	e.ProcessBar(1, bar1, SignalNeutral, 1.0, 0.0)

	pos := e.GetPosition()
	if pos == nil {
		t.Fatal("Should have position")
	}

	// TP/SL should be 0 in signal mode
	if pos.TakeProfit != 0 || pos.StopLoss != 0 {
		t.Errorf("TP/SL should be 0 in signal mode, got TP=%f, SL=%f", pos.TakeProfit, pos.StopLoss)
	}

	// Extreme price movement should not trigger exit
	bar2 := Bar{Open: 100, High: 200, Low: 50, Close: 150, Volume: 1000} // Would hit both TP and SL in TBM
	trade := e.ProcessBar(2, bar2, SignalNeutral, 1.0, 0.0)

	if trade != nil {
		t.Error("Should not exit without opposite signal in signal mode")
	}
}

func TestSignalModeWithRunner(t *testing.T) {
	// Integration test with Runner
	// Next bar exit logic: signal at bar N -> exit at bar N+1 open
	// Reversal logic: after exit, if prevSignal is opposite, enter immediately
	cfg := DefaultConfig()
	cfg.ExitMode = ExitModeSignal
	cfg.InitialCapital = 10000
	cfg.RiskPerTrade = 0.01

	runner, err := NewRunner(cfg)
	if err != nil {
		t.Fatalf("NewRunner failed: %v", err)
	}

	// Create test bars with enough bars for next-bar exits
	bars := []Bar{
		{Open: 100, High: 101, Low: 99, Close: 100, Volume: 1000},   // 0: Long signal
		{Open: 100, High: 105, Low: 99, Close: 104, Volume: 1000},   // 1: Long entry
		{Open: 104, High: 108, Low: 103, Close: 107, Volume: 1000},  // 2: Hold
		{Open: 107, High: 110, Low: 106, Close: 109, Volume: 1000},  // 3: Short signal
		{Open: 109, High: 110, Low: 105, Close: 106, Volume: 1000},  // 4: Long exit + Short entry (reversal)
		{Open: 106, High: 107, Low: 102, Close: 103, Volume: 1000},  // 5: Hold
		{Open: 103, High: 105, Low: 101, Close: 102, Volume: 1000},  // 6: Long signal
		{Open: 102, High: 104, Low: 100, Close: 101, Volume: 1000},  // 7: Short exit + Long entry (reversal)
		{Open: 101, High: 103, Low: 99, Close: 100, Volume: 1000},   // 8: Force close
	}

	signals := []Signal{
		SignalLong,    // 0: Long signal
		SignalNeutral, // 1: Long entry
		SignalNeutral, // 2: Hold
		SignalShort,   // 3: Short signal (Long exit at bar 4, Short entry via reversal)
		SignalNeutral, // 4: Long exit + Short entry
		SignalNeutral, // 5: Hold
		SignalLong,    // 6: Long signal (Short exit at bar 7, Long entry via reversal)
		SignalNeutral, // 7: Short exit + Long entry
		SignalNeutral, // 8: Force close at end
	}

	result := runner.RunWithData(bars, signals, nil, nil)

	// Should have 3 trades: Long, Short (reversal), Long (reversal, force closed)
	if result.TotalTrades != 3 {
		t.Errorf("Expected 3 trades, got %d", result.TotalTrades)
	}

	// First two should exit via signal, last one via timeout (force close)
	if len(result.Trades) >= 2 {
		if result.Trades[0].ExitReason != ExitReasonSignal {
			t.Errorf("Trade 0: exit reason should be SIGNAL, got %s", result.Trades[0].ExitReason)
		}
		if result.Trades[1].ExitReason != ExitReasonSignal {
			t.Errorf("Trade 1: exit reason should be SIGNAL, got %s", result.Trades[1].ExitReason)
		}
	}
	if len(result.Trades) >= 3 {
		if result.Trades[2].ExitReason != ExitReasonTimeout {
			t.Errorf("Trade 2: exit reason should be TIMEOUT (force close), got %s", result.Trades[2].ExitReason)
		}
	}
}

// ============================================================
// Custom Stop Mode Tests
// ============================================================

func TestCustomStopModeLongHit(t *testing.T) {
	// Long position should exit when bar.Low <= stopPrice
	cfg := DefaultConfig()
	cfg.ExitMode = ExitModeCustomStop

	e := NewExecutor(cfg)

	bar0 := Bar{Open: 100, High: 101, Low: 99, Close: 100, Volume: 1000, RealizedVol: 0.02}
	bar1 := Bar{Open: 100, High: 102, Low: 99, Close: 101, Volume: 1000, RealizedVol: 0.02}
	bar2 := Bar{Open: 101, High: 102, Low: 95, Close: 96, Volume: 1000, RealizedVol: 0.02} // Low hits stop

	// Enter long
	e.ProcessBar(0, bar0, SignalLong, 1.0, 0.0)
	e.ProcessBar(1, bar1, SignalNeutral, 1.0, 0.0) // Entry at bar1 Open

	if !e.HasPosition() {
		t.Fatal("Should have position")
	}

	// Stop at 97, bar2 Low is 95 -> should hit
	trade := e.ProcessBar(2, bar2, SignalNeutral, 1.0, 97.0)
	if trade == nil {
		t.Fatal("Should exit due to stop hit")
	}
	if trade.ExitReason != ExitReasonSL {
		t.Errorf("Exit reason should be SL, got %s", trade.ExitReason)
	}
	// Exit price should be at stop price (before slippage)
	if trade.Direction != DirectionLong {
		t.Error("Trade should be long")
	}
}

func TestCustomStopModeShortHit(t *testing.T) {
	// Short position should exit when bar.High >= stopPrice
	cfg := DefaultConfig()
	cfg.ExitMode = ExitModeCustomStop

	e := NewExecutor(cfg)

	bar0 := Bar{Open: 100, High: 101, Low: 99, Close: 100, Volume: 1000, RealizedVol: 0.02}
	bar1 := Bar{Open: 100, High: 101, Low: 98, Close: 99, Volume: 1000, RealizedVol: 0.02}
	bar2 := Bar{Open: 99, High: 105, Low: 98, Close: 104, Volume: 1000, RealizedVol: 0.02} // High hits stop

	// Enter short
	e.ProcessBar(0, bar0, SignalShort, 1.0, 0.0)
	e.ProcessBar(1, bar1, SignalNeutral, 1.0, 0.0) // Entry at bar1 Open

	if !e.HasPosition() {
		t.Fatal("Should have position")
	}

	// Stop at 103, bar2 High is 105 -> should hit
	trade := e.ProcessBar(2, bar2, SignalNeutral, 1.0, 103.0)
	if trade == nil {
		t.Fatal("Should exit due to stop hit")
	}
	if trade.ExitReason != ExitReasonSL {
		t.Errorf("Exit reason should be SL, got %s", trade.ExitReason)
	}
	if trade.Direction != DirectionShort {
		t.Error("Trade should be short")
	}
}

func TestCustomStopModeNoHit(t *testing.T) {
	// Stop price not reached -> position stays open
	// In custom_stop mode, must send SignalLong to maintain position (neutral triggers exit)
	cfg := DefaultConfig()
	cfg.ExitMode = ExitModeCustomStop

	e := NewExecutor(cfg)

	bar0 := Bar{Open: 100, High: 101, Low: 99, Close: 100, Volume: 1000, RealizedVol: 0.02}
	bar1 := Bar{Open: 100, High: 102, Low: 99, Close: 101, Volume: 1000, RealizedVol: 0.02}
	bar2 := Bar{Open: 101, High: 103, Low: 98, Close: 102, Volume: 1000, RealizedVol: 0.02} // Low=98, above stop=95

	// Enter long
	e.ProcessBar(0, bar0, SignalLong, 1.0, 0.0)
	e.ProcessBar(1, bar1, SignalLong, 1.0, 0.0) // Keep Long signal to maintain position

	if !e.HasPosition() {
		t.Fatal("Should have position")
	}

	// Stop at 95, bar2 Low is 98 -> should NOT hit. Keep Long signal.
	trade := e.ProcessBar(2, bar2, SignalLong, 1.0, 95.0)
	if trade != nil {
		t.Error("Should not exit - stop not hit")
	}
	if !e.HasPosition() {
		t.Error("Should still have position")
	}
}

func TestCustomStopModeZeroStop(t *testing.T) {
	// stopPrice=0 means no stop for this bar
	// Must send Long signal to maintain position
	cfg := DefaultConfig()
	cfg.ExitMode = ExitModeCustomStop

	e := NewExecutor(cfg)

	bar0 := Bar{Open: 100, High: 101, Low: 99, Close: 100, Volume: 1000, RealizedVol: 0.02}
	bar1 := Bar{Open: 100, High: 102, Low: 99, Close: 101, Volume: 1000, RealizedVol: 0.02}
	bar2 := Bar{Open: 101, High: 103, Low: 50, Close: 102, Volume: 1000, RealizedVol: 0.02} // Extreme low

	// Enter long
	e.ProcessBar(0, bar0, SignalLong, 1.0, 0.0)
	e.ProcessBar(1, bar1, SignalLong, 1.0, 0.0) // Keep Long signal

	// stopPrice=0 means no stop check
	trade := e.ProcessBar(2, bar2, SignalLong, 1.0, 0.0) // Keep Long signal
	if trade != nil {
		t.Error("Should not exit when stopPrice=0 and maintaining position")
	}
}

func TestCustomStopModeWithSignalExit(t *testing.T) {
	// Custom stop mode exits on neutral OR opposite signal
	cfg := DefaultConfig()
	cfg.ExitMode = ExitModeCustomStop

	e := NewExecutor(cfg)

	bar0 := Bar{Open: 100, High: 101, Low: 99, Close: 100, Volume: 1000, RealizedVol: 0.02}
	bar1 := Bar{Open: 100, High: 102, Low: 99, Close: 101, Volume: 1000, RealizedVol: 0.02}
	bar2 := Bar{Open: 101, High: 103, Low: 100, Close: 102, Volume: 1000, RealizedVol: 0.02}
	bar3 := Bar{Open: 102, High: 104, Low: 101, Close: 103, Volume: 1000, RealizedVol: 0.02}

	// Enter long
	e.ProcessBar(0, bar0, SignalLong, 1.0, 0.0)
	e.ProcessBar(1, bar1, SignalLong, 1.0, 0.0) // Maintain position

	// Neutral signal at bar2 - will trigger exit at bar3
	trade := e.ProcessBar(2, bar2, SignalNeutral, 1.0, 50.0) // Neutral = request exit
	if trade != nil {
		t.Error("Should not exit yet - next bar exit")
	}

	// Next bar: exit due to Neutral signal from previous bar
	trade = e.ProcessBar(3, bar3, SignalNeutral, 1.0, 50.0)
	if trade == nil {
		t.Fatal("Should exit due to neutral signal in custom_stop mode")
	}
	if trade.ExitReason != ExitReasonSignal {
		t.Errorf("Exit reason should be SIGNAL, got %s", trade.ExitReason)
	}
}

func TestCustomStopModeWithRunner(t *testing.T) {
	// Integration test with Runner using custom stop prices
	// In custom_stop mode, send SignalLong to maintain position (neutral = exit)
	cfg := DefaultConfig()
	cfg.ExitMode = ExitModeCustomStop
	cfg.InitialCapital = 10000
	cfg.RiskPerTrade = 0.01

	runner, err := NewRunner(cfg)
	if err != nil {
		t.Fatalf("NewRunner failed: %v", err)
	}

	// Create bars - trade will enter at bar1, hit stop at bar3
	bars := []Bar{
		{Timestamp: 1000, Open: 100, High: 101, Low: 99, Close: 100, Volume: 1000, RealizedVol: 0.02},
		{Timestamp: 2000, Open: 100, High: 102, Low: 99, Close: 101, Volume: 1000, RealizedVol: 0.02},
		{Timestamp: 3000, Open: 101, High: 103, Low: 100, Close: 102, Volume: 1000, RealizedVol: 0.02},
		{Timestamp: 4000, Open: 102, High: 103, Low: 95, Close: 96, Volume: 1000, RealizedVol: 0.02}, // Low hits stop
		{Timestamp: 5000, Open: 96, High: 98, Low: 94, Close: 97, Volume: 1000, RealizedVol: 0.02},
	}

	// Send SignalLong to maintain position, then Neutral after stop hit
	// Bar3: stop is hit, position exits, no new entry (SignalNeutral)
	signals := []Signal{SignalLong, SignalLong, SignalLong, SignalNeutral, SignalNeutral}
	sizes := []float64{1.0, 1.0, 1.0, 1.0, 1.0}
	stopPrices := []float64{0.0, 0.0, 97.0, 97.0, 97.0} // Stop at 97, hit at bar3 (Low=95)

	result := runner.RunWithData(bars, signals, sizes, stopPrices)

	if result.TotalTrades != 1 {
		t.Errorf("Expected 1 trade, got %d", result.TotalTrades)
	}

	if len(result.Trades) > 0 {
		if result.Trades[0].ExitReason != ExitReasonSL {
			t.Errorf("Exit reason should be SL, got %s", result.Trades[0].ExitReason)
		}
		if result.Trades[0].ExitBar != 3 {
			t.Errorf("Should exit at bar 3, got %d", result.Trades[0].ExitBar)
		}
	}
}
