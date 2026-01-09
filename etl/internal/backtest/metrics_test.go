package backtest

import (
	"math"
	"testing"
)

func TestMetricsCalculatorNoTrades(t *testing.T) {
	mc := NewMetricsCalculator(100000, 0.02, false)
	bars := makeTestBars(100, 50000)
	result := mc.Calculate([]Trade{}, bars)

	if result.TotalTrades != 0 {
		t.Errorf("TotalTrades = %d, want 0", result.TotalTrades)
	}
	if len(result.EquityCurve) != 100 {
		t.Errorf("EquityCurve length = %d, want 100", len(result.EquityCurve))
	}
	// All equity should be initial capital
	for i, eq := range result.EquityCurve {
		if eq != 100000 {
			t.Errorf("EquityCurve[%d] = %f, want 100000", i, eq)
		}
	}
}

// makeTestBars creates test bars with given base price
func makeTestBars(count int, basePrice float64) []Bar {
	bars := make([]Bar, count)
	for i := range bars {
		bars[i] = Bar{
			Timestamp: int64(i * 60000),
			Open:      basePrice,
			High:      basePrice * 1.01,
			Low:       basePrice * 0.99,
			Close:     basePrice,
			Volume:    1000,
		}
	}
	return bars
}

func TestMetricsCalculatorWinRate(t *testing.T) {
	mc := NewMetricsCalculator(100000, 0.02, false)

	trades := []Trade{
		{PnL: 0.02, EntryBar: 5, ExitBar: 10, ExitReason: ExitReasonTP, HoldingBars: 5, Direction: DirectionLong, EntryPrice: 50000},
		{PnL: -0.01, EntryBar: 15, ExitBar: 20, ExitReason: ExitReasonSL, HoldingBars: 3, Direction: DirectionLong, EntryPrice: 50000},
		{PnL: 0.03, EntryBar: 22, ExitBar: 30, ExitReason: ExitReasonTP, HoldingBars: 8, Direction: DirectionLong, EntryPrice: 50000},
		{PnL: 0.01, EntryBar: 35, ExitBar: 40, ExitReason: ExitReasonTP, HoldingBars: 4, Direction: DirectionLong, EntryPrice: 50000},
	}

	bars := makeTestBars(50, 50000)
	result := mc.Calculate(trades, bars)

	// 3 wins out of 4
	expectedWinRate := 0.75
	if math.Abs(result.WinRate-expectedWinRate) > 1e-10 {
		t.Errorf("WinRate = %f, want %f", result.WinRate, expectedWinRate)
	}

	// Total trades
	if result.TotalTrades != 4 {
		t.Errorf("TotalTrades = %d, want 4", result.TotalTrades)
	}

	// TP count
	if result.TPCount != 3 {
		t.Errorf("TPCount = %d, want 3", result.TPCount)
	}

	// SL count
	if result.SLCount != 1 {
		t.Errorf("SLCount = %d, want 1", result.SLCount)
	}
}

func TestMetricsCalculatorAvgPnL(t *testing.T) {
	mc := NewMetricsCalculator(100000, 0.02, false)

	trades := []Trade{
		{PnL: 0.02, EntryBar: 5, ExitBar: 10, Direction: DirectionLong, EntryPrice: 50000},
		{PnL: -0.01, EntryBar: 15, ExitBar: 20, Direction: DirectionLong, EntryPrice: 50000},
		{PnL: 0.03, EntryBar: 25, ExitBar: 30, Direction: DirectionLong, EntryPrice: 50000},
	}

	bars := makeTestBars(40, 50000)
	result := mc.Calculate(trades, bars)

	// Avg = (0.02 - 0.01 + 0.03) / 3 = 0.04 / 3
	expectedAvg := 0.04 / 3
	if math.Abs(result.AvgPnL-expectedAvg) > 1e-10 {
		t.Errorf("AvgPnL = %f, want %f", result.AvgPnL, expectedAvg)
	}

	// Total PnL (equity-based with simple interest)
	// positionSize = 100000 * 0.02 = 2000
	// dollarPnL = 2000 * (0.02 - 0.01 + 0.03) = 2000 * 0.04 = 80
	// TotalPnL = 80 / 100000 = 0.0008
	expectedTotal := 0.0008
	if math.Abs(result.TotalPnL-expectedTotal) > 1e-10 {
		t.Errorf("TotalPnL = %f, want %f", result.TotalPnL, expectedTotal)
	}
}

func TestMetricsCalculatorProfitFactor(t *testing.T) {
	mc := NewMetricsCalculator(100000, 0.02, false)
	bars := makeTestBars(30, 50000)

	tests := []struct {
		name   string
		trades []Trade
		want   float64
	}{
		{
			name: "normal case",
			trades: []Trade{
				{PnL: 0.02, EntryBar: 5, ExitBar: 10, Direction: DirectionLong, EntryPrice: 50000},
				{PnL: -0.01, EntryBar: 15, ExitBar: 20, Direction: DirectionLong, EntryPrice: 50000},
				{PnL: 0.03, EntryBar: 22, ExitBar: 28, Direction: DirectionLong, EntryPrice: 50000},
			},
			want: (0.02 + 0.03) / 0.01, // 5.0
		},
		{
			name: "no losses",
			trades: []Trade{
				{PnL: 0.02, EntryBar: 5, ExitBar: 10, Direction: DirectionLong, EntryPrice: 50000},
				{PnL: 0.03, EntryBar: 15, ExitBar: 20, Direction: DirectionLong, EntryPrice: 50000},
			},
			want: math.Inf(1),
		},
		{
			name: "all losses",
			trades: []Trade{
				{PnL: -0.02, EntryBar: 5, ExitBar: 10, Direction: DirectionLong, EntryPrice: 50000},
				{PnL: -0.01, EntryBar: 15, ExitBar: 20, Direction: DirectionLong, EntryPrice: 50000},
			},
			want: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := mc.Calculate(tt.trades, bars)

			if math.IsInf(tt.want, 1) {
				if !math.IsInf(result.ProfitFactor, 1) {
					t.Errorf("ProfitFactor = %f, want Inf", result.ProfitFactor)
				}
			} else if math.Abs(result.ProfitFactor-tt.want) > 1e-10 {
				t.Errorf("ProfitFactor = %f, want %f", result.ProfitFactor, tt.want)
			}
		})
	}
}

func TestMetricsCalculatorAvgHoldBars(t *testing.T) {
	mc := NewMetricsCalculator(100000, 0.02, false)

	trades := []Trade{
		{PnL: 0.02, EntryBar: 5, ExitBar: 10, HoldingBars: 5, Direction: DirectionLong, EntryPrice: 50000},
		{PnL: -0.01, EntryBar: 10, ExitBar: 20, HoldingBars: 10, Direction: DirectionLong, EntryPrice: 50000},
		{PnL: 0.03, EntryBar: 15, ExitBar: 30, HoldingBars: 15, Direction: DirectionLong, EntryPrice: 50000},
	}

	bars := makeTestBars(40, 50000)
	result := mc.Calculate(trades, bars)

	expectedAvg := (5.0 + 10.0 + 15.0) / 3.0
	if math.Abs(result.AvgHoldBars-expectedAvg) > 1e-10 {
		t.Errorf("AvgHoldBars = %f, want %f", result.AvgHoldBars, expectedAvg)
	}
}

func TestCalculateMaxDrawdown(t *testing.T) {
	mc := NewMetricsCalculator(100000, 0.02, false)

	tests := []struct {
		name  string
		curve []float64
		want  float64
	}{
		{
			name:  "no drawdown",
			curve: []float64{100, 110, 120, 130},
			want:  0,
		},
		{
			name:  "single drawdown",
			curve: []float64{100, 90, 95, 100},
			want:  0.10, // 10% drawdown
		},
		{
			name:  "multiple drawdowns",
			curve: []float64{100, 95, 100, 80, 90, 100},
			want:  0.20, // 20% drawdown from 100 to 80
		},
		{
			name:  "new high then drawdown",
			curve: []float64{100, 120, 100, 110},
			want:  20.0 / 120.0, // ~16.67%
		},
		{
			name:  "empty curve",
			curve: []float64{},
			want:  0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := mc.calculateMaxDrawdown(tt.curve)
			if math.Abs(got-tt.want) > 1e-6 {
				t.Errorf("MaxDrawdown = %f, want %f", got, tt.want)
			}
		})
	}
}

func TestCalculateSharpe(t *testing.T) {
	mc := NewMetricsCalculator(100000, 0.02, false)

	// Create bars spanning multiple days (ms per day = 86400000)
	msPerDay := int64(86400000)

	tests := []struct {
		name        string
		equityCurve []float64
		timestamps  []int64 // for creating bars
		wantSign    int     // -1, 0, or 1
	}{
		{
			name:        "positive returns",
			equityCurve: []float64{100000, 101000, 102000, 103000},
			timestamps:  []int64{0, msPerDay, msPerDay * 2, msPerDay * 3},
			wantSign:    1,
		},
		{
			name:        "negative returns",
			equityCurve: []float64{100000, 99000, 98000, 97000},
			timestamps:  []int64{0, msPerDay, msPerDay * 2, msPerDay * 3},
			wantSign:    -1,
		},
		{
			name:        "mixed positive avg",
			equityCurve: []float64{100000, 103000, 102000, 104000},
			timestamps:  []int64{0, msPerDay, msPerDay * 2, msPerDay * 3},
			wantSign:    1,
		},
		{
			name:        "empty",
			equityCurve: []float64{},
			timestamps:  []int64{},
			wantSign:    0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create bars with timestamps
			bars := make([]Bar, len(tt.timestamps))
			for i, ts := range tt.timestamps {
				bars[i] = Bar{Timestamp: ts, Close: 50000}
			}

			got := mc.calculateSharpe(tt.equityCurve, bars)

			var gotSign int
			if got > 0 {
				gotSign = 1
			} else if got < 0 {
				gotSign = -1
			}

			if gotSign != tt.wantSign {
				t.Errorf("Sharpe sign = %d (value=%f), want sign %d", gotSign, got, tt.wantSign)
			}
		})
	}
}

func TestBuildEquityCurve(t *testing.T) {
	mc := NewMetricsCalculator(100000, 0.02, false)

	// Create bars with specific close prices for testing mark-to-market
	bars := make([]Bar, 15)
	for i := range bars {
		bars[i] = Bar{
			Timestamp: int64(i * 60000),
			Open:      50000,
			High:      51000,
			Low:       49000,
			Close:     50000, // Default close price
			Volume:    1000,
		}
	}

	trades := []Trade{
		{
			PnL:        0.10,
			EntryBar:   2,
			ExitBar:    5,
			HoldingBars: 3,
			Direction:  DirectionLong,
			EntryPrice: 50000,
		},
		{
			PnL:        -0.05,
			EntryBar:   8,
			ExitBar:    10,
			HoldingBars: 2,
			Direction:  DirectionLong,
			EntryPrice: 50000,
		},
	}

	curve := mc.buildEquityCurve(trades, bars)

	if len(curve) != 15 {
		t.Fatalf("Curve length = %d, want 15", len(curve))
	}

	// Bars 0-1: Before first trade, should be initial capital
	for i := 0; i < 2; i++ {
		if curve[i] != 100000 {
			t.Errorf("curve[%d] = %f, want 100000", i, curve[i])
		}
	}

	// Bars 2-4: During first trade (unrealized PnL = 0 since close = entry price)
	for i := 2; i < 5; i++ {
		// Close price equals entry price, so unrealized PnL = 0
		if curve[i] != 100000 {
			t.Errorf("curve[%d] = %f, want 100000 (unrealized)", i, curve[i])
		}
	}

	// Bar 5: First trade exit - realized PnL applied (simple interest)
	// positionSize = 100000 * 0.02 = 2000
	// dollarPnL = 2000 * 0.10 = 200
	expectedAfterFirst := 100000 + 2000*0.10 // 100200
	if math.Abs(curve[5]-expectedAfterFirst) > 0.01 {
		t.Errorf("curve[5] = %f, want %f", curve[5], expectedAfterFirst)
	}

	// Bars 6-7: Between trades, should be realized capital
	for i := 6; i < 8; i++ {
		if math.Abs(curve[i]-expectedAfterFirst) > 0.01 {
			t.Errorf("curve[%d] = %f, want %f", i, curve[i], expectedAfterFirst)
		}
	}

	// Bars 8-9: During second trade (unrealized)
	for i := 8; i < 10; i++ {
		// Close = entry price, unrealized PnL = 0
		if math.Abs(curve[i]-expectedAfterFirst) > 0.01 {
			t.Errorf("curve[%d] = %f, want %f (unrealized)", i, curve[i], expectedAfterFirst)
		}
	}

	// Bar 10: Second trade exit (simple interest)
	// dollarPnL = 2000 * (-0.05) = -100
	expectedAfterSecond := expectedAfterFirst + 2000*(-0.05) // 100100
	if math.Abs(curve[10]-expectedAfterSecond) > 0.01 {
		t.Errorf("curve[10] = %f, want %f", curve[10], expectedAfterSecond)
	}

	// Bars 11-14: After second trade
	for i := 11; i < 15; i++ {
		if math.Abs(curve[i]-expectedAfterSecond) > 0.01 {
			t.Errorf("curve[%d] = %f, want %f", i, curve[i], expectedAfterSecond)
		}
	}
}

func TestBuildEquityCurveWithUnrealizedPnL(t *testing.T) {
	mc := NewMetricsCalculator(100000, 0.02, false)

	// Create bars with varying close prices to test unrealized PnL tracking
	bars := make([]Bar, 10)
	for i := range bars {
		closePrice := 50000.0
		if i >= 2 && i < 5 {
			// During trade: price drops by 10% at bar 3, recovers by bar 4
			switch i {
			case 2:
				closePrice = 50000 // Entry bar
			case 3:
				closePrice = 45000 // -10% drawdown
			case 4:
				closePrice = 52000 // Recovery above entry
			}
		}
		bars[i] = Bar{
			Timestamp: int64(i * 60000),
			Open:      50000,
			High:      53000,
			Low:       44000,
			Close:     closePrice,
			Volume:    1000,
		}
	}

	// Trade that enters at bar 2, exits at bar 5 with +4% realized PnL
	trades := []Trade{
		{
			PnL:        0.04, // Final realized PnL
			EntryBar:   2,
			ExitBar:    5,
			HoldingBars: 3,
			Direction:  DirectionLong,
			EntryPrice: 50000,
		},
	}

	curve := mc.buildEquityCurve(trades, bars)

	// Bar 0-1: Before trade - initial capital
	if curve[0] != 100000 || curve[1] != 100000 {
		t.Errorf("Pre-trade equity incorrect: bar0=%f, bar1=%f", curve[0], curve[1])
	}

	// Bar 2: Entry bar, close = 50000, unrealized PnL = 0%
	// Equity = 100000 * (1 + 0 * 0.02) = 100000
	if math.Abs(curve[2]-100000) > 0.01 {
		t.Errorf("curve[2] = %f, want 100000", curve[2])
	}

	// Bar 3: Close = 45000, unrealized PnL = (45000-50000)/50000 = -10%
	// Equity = 100000 * (1 + (-0.10) * 0.02) = 100000 * 0.998 = 99800
	expectedBar3 := 100000 * (1 + (-0.10)*0.02)
	if math.Abs(curve[3]-expectedBar3) > 0.01 {
		t.Errorf("curve[3] = %f, want %f (10%% unrealized loss)", curve[3], expectedBar3)
	}

	// Bar 4: Close = 52000, unrealized PnL = (52000-50000)/50000 = +4%
	// Equity = 100000 * (1 + 0.04 * 0.02) = 100080
	expectedBar4 := 100000 * (1 + 0.04*0.02)
	if math.Abs(curve[4]-expectedBar4) > 0.01 {
		t.Errorf("curve[4] = %f, want %f (4%% unrealized gain)", curve[4], expectedBar4)
	}

	// Bar 5: Exit bar - realized PnL = +4%
	// Equity = 100000 * (1 + 0.04 * 0.02) = 100080
	expectedBar5 := 100000 * (1 + 0.04*0.02)
	if math.Abs(curve[5]-expectedBar5) > 0.01 {
		t.Errorf("curve[5] = %f, want %f (realized)", curve[5], expectedBar5)
	}

	// Bars 6-9: After trade - should maintain realized capital
	for i := 6; i < 10; i++ {
		if math.Abs(curve[i]-expectedBar5) > 0.01 {
			t.Errorf("curve[%d] = %f, want %f", i, curve[i], expectedBar5)
		}
	}
}

func TestBuildEquityCurveMaxDrawdownWithUnrealized(t *testing.T) {
	mc := NewMetricsCalculator(100000, 0.02, false)

	// Test case: MDD should capture unrealized drawdown, not just realized
	// Trade enters at bar 1, has -20% unrealized at bar 2, exits at bar 3 with +2%
	bars := make([]Bar, 5)
	bars[0] = Bar{Close: 50000}
	bars[1] = Bar{Close: 50000}  // Entry bar
	bars[2] = Bar{Close: 40000}  // -20% unrealized drawdown
	bars[3] = Bar{Close: 51000}  // Exit with +2%
	bars[4] = Bar{Close: 51000}

	trades := []Trade{
		{
			PnL:        0.02,
			EntryBar:   1,
			ExitBar:    3,
			Direction:  DirectionLong,
			EntryPrice: 50000,
		},
	}

	curve := mc.buildEquityCurve(trades, bars)

	// Bar 2: -20% unrealized
	// Equity = 100000 * (1 + (-0.20) * 0.02) = 99600
	expectedBar2 := 100000 * (1 + (-0.20)*0.02)
	if math.Abs(curve[2]-expectedBar2) > 0.01 {
		t.Errorf("curve[2] = %f, want %f", curve[2], expectedBar2)
	}

	// Calculate MDD from the curve
	mdd := mc.calculateMaxDrawdown(curve)

	// MDD should be 0.4% (from 100000 to 99600)
	expectedMDD := (100000 - expectedBar2) / 100000
	if math.Abs(mdd-expectedMDD) > 0.0001 {
		t.Errorf("MDD = %f, want %f (should capture unrealized drawdown)", mdd, expectedMDD)
	}
}

func TestBuildEquityCurveShortPosition(t *testing.T) {
	mc := NewMetricsCalculator(100000, 0.02, false)

	// Test short position: unrealized PnL calculation is inverted
	bars := make([]Bar, 5)
	bars[0] = Bar{Close: 50000}
	bars[1] = Bar{Close: 50000}  // Entry bar
	bars[2] = Bar{Close: 45000}  // Price dropped - good for short (+10%)
	bars[3] = Bar{Close: 47500}  // Exit with +5%
	bars[4] = Bar{Close: 47500}

	trades := []Trade{
		{
			PnL:        0.05,
			EntryBar:   1,
			ExitBar:    3,
			Direction:  DirectionShort,
			EntryPrice: 50000,
		},
	}

	curve := mc.buildEquityCurve(trades, bars)

	// Bar 2: Price = 45000, Short unrealized PnL = (50000-45000)/50000 = +10%
	// Equity = 100000 * (1 + 0.10 * 0.02) = 100200
	expectedBar2 := 100000 * (1 + 0.10*0.02)
	if math.Abs(curve[2]-expectedBar2) > 0.01 {
		t.Errorf("curve[2] = %f, want %f (short position +10%% unrealized)", curve[2], expectedBar2)
	}
}

func TestConsecutiveLosses(t *testing.T) {
	tests := []struct {
		name   string
		trades []Trade
		want   int
	}{
		{
			name: "three consecutive losses",
			trades: []Trade{
				{PnL: 0.01},
				{PnL: -0.01},
				{PnL: -0.02},
				{PnL: -0.01},
				{PnL: 0.02},
			},
			want: 3,
		},
		{
			name: "no losses",
			trades: []Trade{
				{PnL: 0.01},
				{PnL: 0.02},
			},
			want: 0,
		},
		{
			name:   "empty",
			trades: []Trade{},
			want:   0,
		},
		{
			name: "all losses",
			trades: []Trade{
				{PnL: -0.01},
				{PnL: -0.02},
				{PnL: -0.03},
			},
			want: 3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ConsecutiveLosses(tt.trades)
			if got != tt.want {
				t.Errorf("ConsecutiveLosses() = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestConsecutiveWins(t *testing.T) {
	trades := []Trade{
		{PnL: 0.01},
		{PnL: 0.02},
		{PnL: 0.03},
		{PnL: -0.01},
		{PnL: 0.01},
	}

	got := ConsecutiveWins(trades)
	if got != 3 {
		t.Errorf("ConsecutiveWins() = %d, want 3", got)
	}
}

func TestSortino(t *testing.T) {
	tests := []struct {
		name     string
		pnls     []float64
		wantSign int
	}{
		{
			name:     "all positive - high sortino",
			pnls:     []float64{0.01, 0.02, 0.03},
			wantSign: 1, // Infinite or very high
		},
		{
			name:     "mixed - positive sortino",
			pnls:     []float64{0.03, -0.01, 0.02, 0.01},
			wantSign: 1,
		},
		{
			name:     "all negative",
			pnls:     []float64{-0.01, -0.02, -0.03},
			wantSign: -1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Sortino(tt.pnls)
			var gotSign int
			if got > 0 {
				gotSign = 1
			} else if got < 0 {
				gotSign = -1
			}
			if gotSign != tt.wantSign {
				t.Errorf("Sortino sign = %d, want %d (value=%f)", gotSign, tt.wantSign, got)
			}
		})
	}
}

func TestCalmar(t *testing.T) {
	tests := []struct {
		name        string
		totalReturn float64
		maxDD       float64
		want        float64
	}{
		{"positive return", 0.20, 0.10, 2.0},
		{"equal return and dd", 0.15, 0.15, 1.0},
		{"zero dd", 0.10, 0, math.Inf(1)},
		{"negative return", -0.10, 0.20, -0.5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Calmar(tt.totalReturn, tt.maxDD)
			if math.IsInf(tt.want, 1) {
				if !math.IsInf(got, 1) {
					t.Errorf("Calmar() = %f, want Inf", got)
				}
			} else if math.Abs(got-tt.want) > 1e-10 {
				t.Errorf("Calmar() = %f, want %f", got, tt.want)
			}
		})
	}
}

func TestExpectancyPerBar(t *testing.T) {
	trades := []Trade{
		{PnL: 0.02, HoldingBars: 5},
		{PnL: -0.01, HoldingBars: 10},
		{PnL: 0.03, HoldingBars: 5},
	}

	got := ExpectancyPerBar(trades)
	// Total PnL = 0.04, Total bars = 20
	expected := 0.04 / 20.0

	if math.Abs(got-expected) > 1e-10 {
		t.Errorf("ExpectancyPerBar() = %f, want %f", got, expected)
	}
}

func TestExpectancyPerBarEmpty(t *testing.T) {
	got := ExpectancyPerBar([]Trade{})
	if got != 0 {
		t.Errorf("ExpectancyPerBar(empty) = %f, want 0", got)
	}
}

func TestExitReasonCounts(t *testing.T) {
	mc := NewMetricsCalculator(100000, 0.02, false)

	trades := []Trade{
		{PnL: 0.02, EntryBar: 5, ExitBar: 10, ExitReason: ExitReasonTP, Direction: DirectionLong, EntryPrice: 50000},
		{PnL: -0.01, EntryBar: 15, ExitBar: 20, ExitReason: ExitReasonSL, Direction: DirectionLong, EntryPrice: 50000},
		{PnL: 0.01, EntryBar: 25, ExitBar: 30, ExitReason: ExitReasonTP, Direction: DirectionLong, EntryPrice: 50000},
		{PnL: 0.00, EntryBar: 35, ExitBar: 40, ExitReason: ExitReasonTimeout, Direction: DirectionLong, EntryPrice: 50000},
		{PnL: -0.02, EntryBar: 45, ExitBar: 50, ExitReason: ExitReasonSL, Direction: DirectionLong, EntryPrice: 50000},
	}

	bars := makeTestBars(60, 50000)
	result := mc.Calculate(trades, bars)

	if result.TPCount != 2 {
		t.Errorf("TPCount = %d, want 2", result.TPCount)
	}
	if result.SLCount != 2 {
		t.Errorf("SLCount = %d, want 2", result.SLCount)
	}
	if result.TimeoutCount != 1 {
		t.Errorf("TimeoutCount = %d, want 1", result.TimeoutCount)
	}
}

// Benchmark
func BenchmarkCalculateMetrics(b *testing.B) {
	mc := NewMetricsCalculator(100000, 0.02, false)

	// Generate 1000 trades
	trades := make([]Trade, 1000)
	for i := range trades {
		entryBar := i * 10
		exitBar := entryBar + 5
		if i%3 == 0 {
			trades[i] = Trade{
				PnL:        -0.01,
				EntryBar:   entryBar,
				ExitBar:    exitBar,
				HoldingBars: 5,
				ExitReason: ExitReasonSL,
				Direction:  DirectionLong,
				EntryPrice: 50000,
			}
		} else {
			trades[i] = Trade{
				PnL:        0.02,
				EntryBar:   entryBar,
				ExitBar:    exitBar + 3,
				HoldingBars: 8,
				ExitReason: ExitReasonTP,
				Direction:  DirectionLong,
				EntryPrice: 50000,
			}
		}
	}

	bars := makeTestBars(10000, 50000)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mc.Calculate(trades, bars)
	}
}

// ============================================================
// Liquidation Tests
// ============================================================

func TestLiquidationOnRealizedLoss(t *testing.T) {
	// Test that equity goes to 0 and stays there after liquidation
	// Using high leverage (riskPerTrade = 1.0) and big loss
	mc := NewMetricsCalculator(10000, 1.0, true) // Full position sizing, compounding

	// Create a trade with 100% loss (ruin scenario with leverage)
	// PnL = -1.0 means 100% loss on the position
	trades := []Trade{
		{
			PnL:        -1.0, // Complete loss
			EntryBar:   5,
			ExitBar:    10,
			HoldingBars: 5,
			ExitReason: ExitReasonSL,
			Direction:  DirectionLong,
			EntryPrice: 50000,
			Size:       1.0,
		},
	}

	bars := makeTestBars(20, 50000)
	result := mc.Calculate(trades, bars)

	// After trade exit at bar 10, equity should be 0
	for i := 10; i < len(result.EquityCurve); i++ {
		if result.EquityCurve[i] != 0 {
			t.Errorf("EquityCurve[%d] = %f, want 0 after liquidation", i, result.EquityCurve[i])
		}
	}

	// Max drawdown should be 100%
	if math.Abs(result.MaxDrawdown-1.0) > 0.01 {
		t.Errorf("MaxDrawdown = %f, want 1.0 (100%%)", result.MaxDrawdown)
	}
}

func TestLiquidationOnUnrealizedLoss(t *testing.T) {
	// Test liquidation during mark-to-market (unrealized loss)
	mc := NewMetricsCalculator(10000, 1.0, true) // Full position, compounding

	// Create a trade that causes unrealized liquidation before exit
	// The trade itself isn't a total loss, but during the position
	// the mark-to-market shows 100%+ loss
	trades := []Trade{
		{
			PnL:        -0.5, // 50% loss on exit
			EntryBar:   5,
			ExitBar:    15,
			HoldingBars: 10,
			ExitReason: ExitReasonSL,
			Direction:  DirectionLong,
			EntryPrice: 100,
			Size:       1.0,
		},
	}

	// Create bars where price drops dramatically during the trade
	bars := make([]Bar, 20)
	for i := range bars {
		price := 100.0
		if i >= 5 && i < 15 {
			// During trade, price crashes
			// At entry (bar 5): price = 100
			// We need unrealized loss > 100% to trigger liquidation
			// For long: unrealized = (current - entry) / entry
			// If current = -10, unrealized = (-10 - 100) / 100 = -1.1 = -110%
			price = 100 - float64(i-5)*20 // Drops by 20 each bar
		}
		bars[i] = Bar{
			Timestamp: int64(i * 60000),
			Open:      price,
			High:      price * 1.01,
			Low:       price * 0.99,
			Close:     price,
			Volume:    1000,
		}
	}

	result := mc.Calculate(trades, bars)

	// Somewhere during the trade, equity should hit 0 due to unrealized loss
	foundZero := false
	for i := 5; i < 15; i++ {
		if result.EquityCurve[i] == 0 {
			foundZero = true
			break
		}
	}

	if !foundZero {
		t.Log("EquityCurve during trade:")
		for i := 5; i < 15; i++ {
			t.Logf("  Bar %d: equity=%f", i, result.EquityCurve[i])
		}
		t.Error("Expected liquidation (equity=0) during unrealized loss period")
	}
}

func TestNoLiquidationOnSmallLoss(t *testing.T) {
	// Normal trading without liquidation
	mc := NewMetricsCalculator(10000, 0.1, false) // 10% position sizing, simple

	trades := []Trade{
		{PnL: -0.1, EntryBar: 5, ExitBar: 10, Direction: DirectionLong, EntryPrice: 100, Size: 1.0},
		{PnL: 0.2, EntryBar: 15, ExitBar: 20, Direction: DirectionLong, EntryPrice: 100, Size: 1.0},
	}

	bars := makeTestBars(30, 100)
	result := mc.Calculate(trades, bars)

	// No equity should be 0
	for i, eq := range result.EquityCurve {
		if eq == 0 {
			t.Errorf("EquityCurve[%d] = 0, but should not be liquidated with small losses", i)
		}
	}

	// Final equity should be positive
	finalEquity := result.EquityCurve[len(result.EquityCurve)-1]
	if finalEquity <= 0 {
		t.Errorf("Final equity = %f, want positive", finalEquity)
	}
}

func TestLiquidationStaysAtZero(t *testing.T) {
	// After liquidation, equity should stay at 0 even with subsequent "trades"
	mc := NewMetricsCalculator(10000, 1.0, true)

	trades := []Trade{
		{
			PnL:        -1.0, // Liquidation
			EntryBar:   5,
			ExitBar:    10,
			Direction:  DirectionLong,
			EntryPrice: 100,
			Size:       1.0,
		},
		{
			PnL:        0.5, // This trade shouldn't matter - already liquidated
			EntryBar:   15,
			ExitBar:    20,
			Direction:  DirectionLong,
			EntryPrice: 100,
			Size:       1.0,
		},
	}

	bars := makeTestBars(30, 100)
	result := mc.Calculate(trades, bars)

	// All bars after liquidation should be 0
	for i := 10; i < len(result.EquityCurve); i++ {
		if result.EquityCurve[i] != 0 {
			t.Errorf("EquityCurve[%d] = %f, want 0 (should stay liquidated)", i, result.EquityCurve[i])
		}
	}
}

func TestLiquidationWithLeverage(t *testing.T) {
	// Test liquidation with leveraged position (Size > 1)
	mc := NewMetricsCalculator(10000, 0.1, true) // 10% base, but size = 10 = 100% effective

	trades := []Trade{
		{
			PnL:        -0.1, // 10% loss, but with 10x size = 100% loss
			EntryBar:   5,
			ExitBar:    10,
			Direction:  DirectionLong,
			EntryPrice: 100,
			Size:       10.0, // 10x leverage
		},
	}

	bars := makeTestBars(20, 100)
	result := mc.Calculate(trades, bars)

	// With compounding: capital *= (1 + PnL * riskPerTrade * size)
	// = 10000 * (1 + (-0.1) * 0.1 * 10) = 10000 * (1 - 0.1) = 9000
	// Wait, that's not liquidation. Let me recalculate.
	// Actually: 1 + (-0.1) * 0.1 * 10 = 1 - 0.1 = 0.9
	// So capital = 10000 * 0.9 = 9000, not liquidated.
	//
	// For liquidation: 1 + PnL * riskPerTrade * size <= 0
	// PnL * riskPerTrade * size <= -1
	// With riskPerTrade=0.1, size=10: PnL <= -1
	// So need PnL = -1 for liquidation

	// Actually this test case won't trigger liquidation with current params
	// Let me verify the final equity is 9000 (not liquidated)
	finalEquity := result.EquityCurve[len(result.EquityCurve)-1]
	expectedEquity := 10000.0 * (1 + (-0.1)*0.1*10.0) // = 9000

	if math.Abs(finalEquity-expectedEquity) > 0.01 {
		t.Errorf("Final equity = %f, want %f", finalEquity, expectedEquity)
	}

	// Now test with actual liquidation scenario
	mc2 := NewMetricsCalculator(10000, 0.1, true)
	trades2 := []Trade{
		{
			PnL:        -1.0, // 100% loss with leverage = liquidation
			EntryBar:   5,
			ExitBar:    10,
			Direction:  DirectionLong,
			EntryPrice: 100,
			Size:       10.0,
		},
	}

	result2 := mc2.Calculate(trades2, bars)

	// After this: capital *= (1 + (-1.0) * 0.1 * 10) = capital * 0 = 0
	for i := 10; i < len(result2.EquityCurve); i++ {
		if result2.EquityCurve[i] != 0 {
			t.Errorf("EquityCurve[%d] = %f, want 0 (liquidated with leverage)", i, result2.EquityCurve[i])
		}
	}
}
