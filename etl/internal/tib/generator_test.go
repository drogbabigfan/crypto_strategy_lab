package tib

import (
	"math"
	"testing"
)

// Mock Trade Struct for testing
type Trade struct {
	ID           int64
	Price        float64
	Quantity     float64
	Time         int64 // Unix
	IsBuyerMaker bool
}

func TestTickRule(t *testing.T) {
	// Setup
	gen := NewTIBGenerator(TIBConfig{
		ExpectedTickCount: 100, // High enough not to trigger bar
		ProbBuy:           0.5,
	})

	tests := []struct {
		price    float64
		expected int // 1 or -1
	}{
		{100.0, 1},  // First tick: prevPrice=0 -> 100 > 0 -> Up
		{101.0, 1},  // Up: 101 > 100
		{101.0, 1},  // Flat: keep prev direction (1)
		{100.5, -1}, // Down: 100.5 < 101
		{100.5, -1}, // Flat: keep prev direction (-1)
		{102.0, 1},  // Up: 102 > 100.5
	}

	for i, tt := range tests {
		// Use ProcessTrade to properly update state (prevPrice, prevDir)
		// We pass quantity=0 since we only care about tick rule
		gen.ProcessTrade(tt.price, 0, int64(i))

		// After ProcessTrade, prevDir should be updated
		if gen.prevDir != tt.expected {
			t.Errorf("Step %d: Price %.2f, Expected direction %d, Got %d", i, tt.price, tt.expected, gen.prevDir)
		}
	}
}

func TestBarGeneration(t *testing.T) {
	// Scenario:
	// E_T = 5
	// ProbBuy = 0.9
	// |2*0.9 - 1| = 0.8
	// Threshold = 5 * 0.8 = 4.0

	config := TIBConfig{
		ExpectedTickCount: 5,
		ProbBuy:           0.9,
	}
	gen := NewTIBGenerator(config)

	// Input: Sequence of UP ticks
	// T0: 100. prevPrice=0 -> Up. imbalance=1. |1| < 4. No bar.
	// T1: 101. Up. imbalance=2. |2| < 4. No bar.
	// T2: 102. Up. imbalance=3. |3| < 4. No bar.
	// T3: 103. Up. imbalance=4. |4| >= 4. -> TRIGGER BAR. Close=103.
	// T4: 104. Up. imbalance=1 (new bar). |1| < 4. No bar.
	// T5: 103. Down. imbalance=0. |0| < 4. No bar.

	trades := []Trade{
		{Price: 100.0, Time: 1000},
		{Price: 101.0, Time: 1001},
		{Price: 102.0, Time: 1002},
		{Price: 103.0, Time: 1003}, // Bar triggered here
		{Price: 104.0, Time: 1004},
		{Price: 103.0, Time: 1005},
	}

	var bars []Bar

	for _, tr := range trades {
		if bar := gen.ProcessTrade(tr.Price, tr.Quantity, tr.Time); bar != nil {
			bars = append(bars, *bar)
		}
	}

	if len(bars) != 1 {
		t.Fatalf("Expected 1 bar, got %d", len(bars))
	}

	// Verify Bar Content
	b := bars[0]

	// Close should be 103.0 (the price at which threshold was reached)
	if b.Close != 103.0 {
		t.Errorf("Expected Close 103.0, got %.2f", b.Close)
	}

	// Open should be 100.0 (first price in the bar)
	if b.Open != 100.0 {
		t.Errorf("Expected Open 100.0, got %.2f", b.Open)
	}

	// High should be 103.0
	if b.High != 103.0 {
		t.Errorf("Expected High 103.0, got %.2f", b.High)
	}

	// Low should be 100.0
	if b.Low != 100.0 {
		t.Errorf("Expected Low 100.0, got %.2f", b.Low)
	}

	// TickCount should be 4 (T0, T1, T2, T3)
	if b.TickCount != 4 {
		t.Errorf("Expected TickCount 4, got %d", b.TickCount)
	}
}

func TestPrecisionRounding(t *testing.T) {
	// Verify that stored values are rounded to 8 decimals
	// e.g. 100.123456789 -> 100.12345679

	val := 100.123456789
	rounded := Round8(val)
	expected := 100.12345679

	if math.Abs(rounded-expected) > 1e-10 {
		t.Errorf("Rounding failed. Input %.9f, Got %.9f, Expected %.9f", val, rounded, expected)
	}
}

func TestMultipleBars(t *testing.T) {
	// Test that multiple bars are generated correctly
	config := TIBConfig{
		ExpectedTickCount: 3,
		ProbBuy:           1.0, // threshold = 3 * |2*1 - 1| = 3 * 1 = 3
	}
	gen := NewTIBGenerator(config)

	// All up ticks - should generate bar every 3 ticks
	prices := []float64{100, 101, 102, 103, 104, 105, 106, 107, 108}
	var bars []Bar

	for i, p := range prices {
		if bar := gen.ProcessTrade(p, 1.0, int64(i*1000)); bar != nil {
			bars = append(bars, *bar)
		}
	}

	// Expected: bars at tick 3 (102), 6 (105), 9 (108)
	if len(bars) != 3 {
		t.Fatalf("Expected 3 bars, got %d", len(bars))
	}

	expectedCloses := []float64{102, 105, 108}
	for i, b := range bars {
		if b.Close != expectedCloses[i] {
			t.Errorf("Bar %d: Expected Close %.1f, got %.2f", i, expectedCloses[i], b.Close)
		}
	}
}

func TestImbalanceReset(t *testing.T) {
	// Test that imbalance resets after bar generation
	config := TIBConfig{
		ExpectedTickCount: 2,
		ProbBuy:           1.0, // threshold = 2
	}
	gen := NewTIBGenerator(config)

	// Two up ticks -> bar, then two down ticks -> bar
	trades := []struct {
		price float64
	}{
		{100}, // imbalance = 1
		{101}, // imbalance = 2 -> bar
		{100}, // imbalance = -1 (reset, then -1)
		{99},  // imbalance = -2 -> bar
	}

	var bars []Bar
	for i, tr := range trades {
		if bar := gen.ProcessTrade(tr.price, 1.0, int64(i)); bar != nil {
			bars = append(bars, *bar)
		}
	}

	if len(bars) != 2 {
		t.Fatalf("Expected 2 bars, got %d", len(bars))
	}

	// First bar: up trend
	if bars[0].Close != 101 {
		t.Errorf("First bar: Expected Close 101, got %.2f", bars[0].Close)
	}

	// Second bar: down trend
	if bars[1].Close != 99 {
		t.Errorf("Second bar: Expected Close 99, got %.2f", bars[1].Close)
	}
}

func TestStateExportImport(t *testing.T) {
	// Test that exporting and importing state preserves generator behavior
	config := TIBConfig{
		ExpectedTickCount: 5,
		ProbBuy:           0.9, // threshold = 4
	}

	gen1 := NewTIBGenerator(config)

	// Process some trades to build up state
	trades := []struct {
		price float64
		time  int64
	}{
		{100.0, 1000},
		{101.0, 1001},
		{102.0, 1002}, // imbalance = 3, no bar yet
	}

	for _, tr := range trades {
		gen1.ProcessTrade(tr.price, 1.0, tr.time)
	}

	// Export state
	state := gen1.ExportState()

	// Verify exported state
	if state.PrevPrice != 102.0 {
		t.Errorf("Exported PrevPrice: expected 102.0, got %.2f", state.PrevPrice)
	}
	if state.PrevDir != 1 {
		t.Errorf("Exported PrevDir: expected 1, got %d", state.PrevDir)
	}
	if state.Imbalance != 3 {
		t.Errorf("Exported Imbalance: expected 3, got %.2f", state.Imbalance)
	}
	if state.CurrentBar == nil {
		t.Fatal("Exported CurrentBar should not be nil")
	}

	// Create new generator and import state
	gen2 := NewTIBGenerator(config)
	gen2.ImportState(state)

	// Continue processing - should trigger bar at next up tick
	bar := gen2.ProcessTrade(103.0, 1.0, 1003)

	if bar == nil {
		t.Fatal("Expected bar to be triggered after state import")
	}

	// Bar should include all 4 ticks (100, 101, 102, 103)
	if bar.Open != 100.0 {
		t.Errorf("Bar Open: expected 100.0, got %.2f", bar.Open)
	}
	if bar.Close != 103.0 {
		t.Errorf("Bar Close: expected 103.0, got %.2f", bar.Close)
	}
	if bar.TickCount != 4 {
		t.Errorf("Bar TickCount: expected 4, got %d", bar.TickCount)
	}
}

func TestStateContinuityAcrossDataGaps(t *testing.T) {
	// Simulate processing data with a gap (e.g., month boundary)
	// State should carry over correctly
	config := TIBConfig{
		ExpectedTickCount: 4,
		ProbBuy:           1.0, // threshold = 4
	}

	gen := NewTIBGenerator(config)

	// === First batch (e.g., January data) ===
	batch1 := []struct {
		price float64
		time  int64
	}{
		{100.0, 1000},
		{101.0, 1001}, // imbalance = 2
	}

	for _, tr := range batch1 {
		gen.ProcessTrade(tr.price, 1.0, tr.time)
	}

	// Export state (simulate saving at end of month)
	savedState := gen.ExportState()

	// === Simulate restart - create new generator ===
	gen2 := NewTIBGenerator(config)
	gen2.ImportState(savedState)

	// === Second batch (e.g., February data) ===
	// Data has a time gap but state is continuous
	batch2 := []struct {
		price float64
		time  int64
	}{
		{102.0, 5000}, // Big time gap, but state continues. imbalance = 3
		{103.0, 5001}, // imbalance = 4 -> bar triggered
	}

	var bars []Bar
	for _, tr := range batch2 {
		if bar := gen2.ProcessTrade(tr.price, 1.0, tr.time); bar != nil {
			bars = append(bars, *bar)
		}
	}

	if len(bars) != 1 {
		t.Fatalf("Expected 1 bar, got %d", len(bars))
	}

	// The bar should span across the gap
	// Open from batch1, Close from batch2
	if bars[0].Open != 100.0 {
		t.Errorf("Bar Open: expected 100.0 (from batch1), got %.2f", bars[0].Open)
	}
	if bars[0].Close != 103.0 {
		t.Errorf("Bar Close: expected 103.0 (from batch2), got %.2f", bars[0].Close)
	}
	if bars[0].TickCount != 4 {
		t.Errorf("Bar TickCount: expected 4 (2 from batch1 + 2 from batch2), got %d", bars[0].TickCount)
	}
}

func TestStateWithPendingBar(t *testing.T) {
	// Test that incomplete bar is preserved across state save/load
	config := TIBConfig{
		ExpectedTickCount: 100,
		ProbBuy:           0.9, // threshold = 100 * 0.8 = 80 (high enough to not trigger)
	}

	gen1 := NewTIBGenerator(config)

	// Process trades but don't complete a bar
	gen1.ProcessTrade(100.0, 1.5, 1000)
	gen1.ProcessTrade(101.0, 2.5, 1001)
	gen1.ProcessTrade(100.5, 3.0, 1002)

	// Export and import
	state := gen1.ExportState()
	gen2 := NewTIBGenerator(config)
	gen2.ImportState(state)

	// Verify the pending bar state is preserved
	if state.CurrentBar == nil {
		t.Fatal("CurrentBar should be preserved in state")
	}
	if state.CurrentBar.Open != 100.0 {
		t.Errorf("Pending bar Open: expected 100.0, got %.2f", state.CurrentBar.Open)
	}
	if state.CurrentBar.High != 101.0 {
		t.Errorf("Pending bar High: expected 101.0, got %.2f", state.CurrentBar.High)
	}
	if state.CurrentBar.Low != 100.0 {
		t.Errorf("Pending bar Low: expected 100.0, got %.2f", state.CurrentBar.Low)
	}
	if state.CurrentBar.Volume != 7.0 { // 1.5 + 2.5 + 3.0
		t.Errorf("Pending bar Volume: expected 7.0, got %.2f", state.CurrentBar.Volume)
	}
	if state.CurrentBar.TickCount != 3 {
		t.Errorf("Pending bar TickCount: expected 3, got %d", state.CurrentBar.TickCount)
	}
}
