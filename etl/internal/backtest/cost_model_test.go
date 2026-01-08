package backtest

import (
	"math"
	"testing"
)

func TestNewCostModel(t *testing.T) {
	cfg := DefaultConfig()
	cm := NewCostModel(cfg)

	if cm.BaseFee != cfg.BaseFee {
		t.Errorf("BaseFee mismatch: got %f, want %f", cm.BaseFee, cfg.BaseFee)
	}
	if cm.ImpactCoeff != cfg.ImpactCoeff {
		t.Errorf("ImpactCoeff mismatch: got %f, want %f", cm.ImpactCoeff, cfg.ImpactCoeff)
	}
}

func TestGetSlippage(t *testing.T) {
	tests := []struct {
		name       string
		volatility float64
		tradeSize  float64
		avgVolume  float64
		wantMin    float64
		wantMax    float64
	}{
		{
			name:       "zero volume returns cap",
			volatility: 0.02,
			tradeSize:  1000,
			avgVolume:  0,
			wantMin:    0.005, // SlippageCap
			wantMax:    0.005,
		},
		{
			name:       "negative volume returns cap",
			volatility: 0.02,
			tradeSize:  1000,
			avgVolume:  -100,
			wantMin:    0.005,
			wantMax:    0.005,
		},
		{
			name:       "small trade low impact",
			volatility: 0.02,
			tradeSize:  100,
			avgVolume:  10000,
			wantMin:    0.0001, // At least BaseSlippage
			wantMax:    0.001,  // Should be small
		},
		{
			name:       "large trade high impact",
			volatility: 0.05,
			tradeSize:  5000,
			avgVolume:  10000,
			wantMin:    0.003,
			wantMax:    0.005, // Capped
		},
		{
			name:       "very large trade capped",
			volatility: 0.10,
			tradeSize:  10000,
			avgVolume:  1000,
			wantMin:    0.005, // Should be capped
			wantMax:    0.005,
		},
	}

	cm := DefaultCostModel()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := cm.GetSlippage(tt.volatility, tt.tradeSize, tt.avgVolume)
			if got < tt.wantMin || got > tt.wantMax {
				t.Errorf("GetSlippage() = %f, want in [%f, %f]", got, tt.wantMin, tt.wantMax)
			}
		})
	}
}

func TestGetSlippageFormula(t *testing.T) {
	// Verify the Square Root Law formula
	cm := &CostModel{
		BaseFee:      0.001,
		BaseSlippage: 0.0001,
		ImpactCoeff:  0.1,
		SlippageCap:  0.01, // High cap to test formula
	}

	volatility := 0.02
	tradeSize := 100.0
	avgVolume := 10000.0

	// Expected: 0.0001 + 0.1 * 0.02 * sqrt(100/10000)
	// = 0.0001 + 0.1 * 0.02 * 0.1
	// = 0.0001 + 0.0002
	// = 0.0003
	expected := 0.0001 + 0.1*0.02*math.Sqrt(100.0/10000.0)
	got := cm.GetSlippage(volatility, tradeSize, avgVolume)

	if math.Abs(got-expected) > 1e-10 {
		t.Errorf("GetSlippage formula: got %f, want %f", got, expected)
	}
}

func TestGetTotalCost(t *testing.T) {
	cm := DefaultCostModel()

	volatility := 0.02
	tradeSize := 100.0
	avgVolume := 10000.0

	slippage := cm.GetSlippage(volatility, tradeSize, avgVolume)
	expectedCost := 2 * (cm.BaseFee + slippage)

	got := cm.GetTotalCost(volatility, tradeSize, avgVolume)

	if math.Abs(got-expectedCost) > 1e-10 {
		t.Errorf("GetTotalCost() = %f, want %f", got, expectedCost)
	}
}

func TestApplyEntrySlippage(t *testing.T) {
	cm := DefaultCostModel()

	price := 50000.0
	volatility := 0.02
	tradeSize := 100.0
	avgVolume := 10000.0
	slippage := cm.GetSlippage(volatility, tradeSize, avgVolume)

	t.Run("long pays more", func(t *testing.T) {
		got := cm.ApplyEntrySlippage(price, DirectionLong, volatility, tradeSize, avgVolume)
		expected := price * (1 + slippage)

		if math.Abs(got-expected) > 1e-6 {
			t.Errorf("Long entry: got %f, want %f", got, expected)
		}
		if got <= price {
			t.Error("Long entry should be higher than base price")
		}
	})

	t.Run("short receives less", func(t *testing.T) {
		got := cm.ApplyEntrySlippage(price, DirectionShort, volatility, tradeSize, avgVolume)
		expected := price * (1 - slippage)

		if math.Abs(got-expected) > 1e-6 {
			t.Errorf("Short entry: got %f, want %f", got, expected)
		}
		if got >= price {
			t.Error("Short entry should be lower than base price")
		}
	})
}

func TestApplyExitSlippage(t *testing.T) {
	cm := DefaultCostModel()

	price := 50000.0
	volatility := 0.02
	tradeSize := 100.0
	avgVolume := 10000.0
	slippage := cm.GetSlippage(volatility, tradeSize, avgVolume)

	t.Run("long receives less", func(t *testing.T) {
		got := cm.ApplyExitSlippage(price, DirectionLong, volatility, tradeSize, avgVolume)
		expected := price * (1 - slippage)

		if math.Abs(got-expected) > 1e-6 {
			t.Errorf("Long exit: got %f, want %f", got, expected)
		}
		if got >= price {
			t.Error("Long exit should be lower than base price")
		}
	})

	t.Run("short pays more", func(t *testing.T) {
		got := cm.ApplyExitSlippage(price, DirectionShort, volatility, tradeSize, avgVolume)
		expected := price * (1 + slippage)

		if math.Abs(got-expected) > 1e-6 {
			t.Errorf("Short exit: got %f, want %f", got, expected)
		}
		if got <= price {
			t.Error("Short exit should be higher than base price")
		}
	})
}

func TestCalculatePnL(t *testing.T) {
	cm := &CostModel{
		BaseFee:      0.001, // 0.1%
		BaseSlippage: 0,
		ImpactCoeff:  0,
		SlippageCap:  0.01,
	}

	tests := []struct {
		name         string
		direction    Direction
		entryPrice   float64
		exitPrice    float64
		wantGrossPnL float64
		wantNetPnL   float64
	}{
		{
			name:         "long profit",
			direction:    DirectionLong,
			entryPrice:   100,
			exitPrice:    110,
			wantGrossPnL: 0.10,           // 10% gain
			wantNetPnL:   0.10 - 0.002,   // minus 0.2% fees (2 × 0.1%)
		},
		{
			name:         "long loss",
			direction:    DirectionLong,
			entryPrice:   100,
			exitPrice:    95,
			wantGrossPnL: -0.05,          // 5% loss
			wantNetPnL:   -0.05 - 0.002,  // minus fees
		},
		{
			name:         "short profit",
			direction:    DirectionShort,
			entryPrice:   100,
			exitPrice:    90,
			wantGrossPnL: 0.10,           // 10% gain (price dropped)
			wantNetPnL:   0.10 - 0.002,
		},
		{
			name:         "short loss",
			direction:    DirectionShort,
			entryPrice:   100,
			exitPrice:    105,
			wantGrossPnL: -0.05,          // 5% loss (price rose)
			wantNetPnL:   -0.05 - 0.002,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotGross, gotNet := cm.CalculatePnL(tt.entryPrice, tt.exitPrice, tt.direction)

			if math.Abs(gotGross-tt.wantGrossPnL) > 1e-10 {
				t.Errorf("GrossPnL = %f, want %f", gotGross, tt.wantGrossPnL)
			}
			if math.Abs(gotNet-tt.wantNetPnL) > 1e-10 {
				t.Errorf("NetPnL = %f, want %f", gotNet, tt.wantNetPnL)
			}
		})
	}
}

func TestIsViable(t *testing.T) {
	cm := DefaultCostModel()

	tests := []struct {
		name       string
		ptMult     float64
		volatility float64
		tradeSize  float64
		avgVolume  float64
		buffer     float64
		want       bool
	}{
		{
			name:       "high vol trade viable",
			ptMult:     2.0,
			volatility: 0.05,   // 5% vol
			tradeSize:  100,
			avgVolume:  10000,
			buffer:     1.5,
			want:       true,   // 2 * 0.05 = 0.10 >> costs
		},
		{
			name:       "low vol trade not viable",
			ptMult:     1.5,
			volatility: 0.002,  // 0.2% vol
			tradeSize:  100,
			avgVolume:  10000,
			buffer:     1.5,
			want:       false,  // 1.5 * 0.002 = 0.003 < costs
		},
		{
			name:       "large trade high impact still viable with high PT",
			ptMult:     2.0,
			volatility: 0.02,
			tradeSize:  10000,  // Large trade
			avgVolume:  1000,   // Low volume
			buffer:     1.5,
			want:       true,   // Expected: 4% > Required: 1.8%
		},
		{
			name:       "tiny PT not viable",
			ptMult:     0.5,
			volatility: 0.01,   // 1% vol
			tradeSize:  1000,
			avgVolume:  1000,
			buffer:     1.5,
			want:       false,  // Expected: 0.5% < Required: 0.63%
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := cm.IsViable(tt.ptMult, tt.volatility, tt.tradeSize, tt.avgVolume, tt.buffer)
			if got != tt.want {
				t.Errorf("IsViable() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestSlippageSymmetry(t *testing.T) {
	// Slippage should always hurt the trader
	cm := DefaultCostModel()

	price := 50000.0
	volatility := 0.02
	tradeSize := 100.0
	avgVolume := 10000.0

	// Long: entry high, exit low → always loses to slippage
	longEntry := cm.ApplyEntrySlippage(price, DirectionLong, volatility, tradeSize, avgVolume)
	longExit := cm.ApplyExitSlippage(price, DirectionLong, volatility, tradeSize, avgVolume)

	if longEntry <= price {
		t.Error("Long entry should be above market price")
	}
	if longExit >= price {
		t.Error("Long exit should be below market price")
	}

	// Short: entry low, exit high → always loses to slippage
	shortEntry := cm.ApplyEntrySlippage(price, DirectionShort, volatility, tradeSize, avgVolume)
	shortExit := cm.ApplyExitSlippage(price, DirectionShort, volatility, tradeSize, avgVolume)

	if shortEntry >= price {
		t.Error("Short entry should be below market price")
	}
	if shortExit <= price {
		t.Error("Short exit should be above market price")
	}
}

func TestCostModelEdgeCases(t *testing.T) {
	cm := DefaultCostModel()

	t.Run("zero volatility", func(t *testing.T) {
		slippage := cm.GetSlippage(0, 100, 10000)
		// Should be BaseSlippage only
		if slippage != cm.BaseSlippage {
			t.Errorf("Zero volatility slippage = %f, want %f", slippage, cm.BaseSlippage)
		}
	})

	t.Run("zero trade size", func(t *testing.T) {
		slippage := cm.GetSlippage(0.02, 0, 10000)
		// Should be BaseSlippage only
		if slippage != cm.BaseSlippage {
			t.Errorf("Zero trade size slippage = %f, want %f", slippage, cm.BaseSlippage)
		}
	})

	t.Run("equal prices zero pnl", func(t *testing.T) {
		grossPnL, _ := cm.CalculatePnL(100, 100, DirectionLong)
		if grossPnL != 0 {
			t.Errorf("Same entry/exit should have zero gross PnL, got %f", grossPnL)
		}
	})
}

// Benchmark tests
func BenchmarkGetSlippage(b *testing.B) {
	cm := DefaultCostModel()
	for i := 0; i < b.N; i++ {
		cm.GetSlippage(0.02, 100, 10000)
	}
}

func BenchmarkCalculatePnL(b *testing.B) {
	cm := DefaultCostModel()
	for i := 0; i < b.N; i++ {
		cm.CalculatePnL(50000, 51000, DirectionLong)
	}
}
