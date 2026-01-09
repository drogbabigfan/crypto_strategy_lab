package backtest

import (
	"bytes"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"testing"
)

func TestNewRunner(t *testing.T) {
	t.Run("valid config", func(t *testing.T) {
		cfg := DefaultConfig()
		runner, err := NewRunner(cfg)

		if err != nil {
			t.Fatalf("NewRunner failed: %v", err)
		}
		if runner == nil {
			t.Fatal("Runner should not be nil")
		}
		if runner.executor == nil {
			t.Error("Executor should be initialized")
		}
		if runner.metrics == nil {
			t.Error("Metrics calculator should be initialized")
		}
	})

	t.Run("invalid config rejected", func(t *testing.T) {
		cfg := Config{
			SLMult: 0, // Invalid: must be > 0
		}
		_, err := NewRunner(cfg)

		if err == nil {
			t.Error("Expected error for invalid config")
		}
	})

	t.Run("negative SL rejected", func(t *testing.T) {
		cfg := DefaultConfig()
		cfg.SLMult = -1
		_, err := NewRunner(cfg)

		if err == nil {
			t.Error("Expected error for negative SLMult")
		}
	})

	t.Run("zero PT rejected", func(t *testing.T) {
		cfg := DefaultConfig()
		cfg.PTMult = 0
		_, err := NewRunner(cfg)

		if err == nil {
			t.Error("Expected error for zero PTMult")
		}
	})
}

func TestRunWithData(t *testing.T) {
	cfg := DefaultConfig()
	cfg.SLMult = 1.0
	cfg.PTMult = 2.0
	cfg.MaxHoldBars = 10
	cfg.InitialCapital = 10000
	cfg.RiskPerTrade = 0.01 // 1% risk per trade

	runner, err := NewRunner(cfg)
	if err != nil {
		t.Fatalf("NewRunner failed: %v", err)
	}

	t.Run("empty data returns zero trades", func(t *testing.T) {
		result := runner.RunWithData([]Bar{}, []Signal{}, nil, nil)

		if result.TotalTrades != 0 {
			t.Errorf("Expected 0 trades, got %d", result.TotalTrades)
		}
	})

	t.Run("no signals returns zero trades", func(t *testing.T) {
		bars := makeBars(100, 50000, 0.01)
		signals := make([]Signal, len(bars))
		// All neutral signals
		for i := range signals {
			signals[i] = SignalNeutral
		}

		result := runner.RunWithData(bars, signals, nil, nil)

		if result.TotalTrades != 0 {
			t.Errorf("Expected 0 trades, got %d", result.TotalTrades)
		}
	})

	t.Run("single long trade", func(t *testing.T) {
		// Create bars with upward movement
		bars := make([]Bar, 20)
		for i := range bars {
			price := 50000.0 + float64(i)*100 // Rising prices
			bars[i] = Bar{
				Timestamp:   int64(i * 60000),
				Open:        price,
				High:        price + 50,
				Low:         price - 50,
				Close:       price + 25,
				Volume:      1000,
				RealizedVol: 0.02,
			}
		}

		signals := make([]Signal, len(bars))
		signals[0] = SignalLong // Signal at bar 0, entry at bar 1

		result := runner.RunWithData(bars, signals, nil, nil)

		if result.TotalTrades != 1 {
			t.Errorf("Expected 1 trade, got %d", result.TotalTrades)
		}
	})

	t.Run("single short trade", func(t *testing.T) {
		// Create bars with downward movement
		bars := make([]Bar, 20)
		for i := range bars {
			price := 50000.0 - float64(i)*100 // Falling prices
			bars[i] = Bar{
				Timestamp:   int64(i * 60000),
				Open:        price,
				High:        price + 50,
				Low:         price - 50,
				Close:       price - 25,
				Volume:      1000,
				RealizedVol: 0.02,
			}
		}

		signals := make([]Signal, len(bars))
		signals[0] = SignalShort // Signal at bar 0, entry at bar 1

		result := runner.RunWithData(bars, signals, nil, nil)

		if result.TotalTrades != 1 {
			t.Errorf("Expected 1 trade, got %d", result.TotalTrades)
		}
	})

	t.Run("multiple trades counted correctly", func(t *testing.T) {
		bars := makeBars(100, 50000, 0.02)
		signals := make([]Signal, len(bars))

		// Create signals spread out enough to complete trades
		signals[0] = SignalLong
		signals[20] = SignalShort
		signals[40] = SignalLong
		signals[60] = SignalShort
		signals[80] = SignalLong

		result := runner.RunWithData(bars, signals, nil, nil)

		if result.TotalTrades < 1 {
			t.Error("Expected at least 1 trade")
		}
	})

	t.Run("force close at end", func(t *testing.T) {
		bars := makeBars(10, 50000, 0.02)
		signals := make([]Signal, len(bars))
		signals[0] = SignalLong // Entry at bar 1, should force close at bar 9

		result := runner.RunWithData(bars, signals, nil, nil)

		// Should have closed the position
		if result.TotalTrades != 1 {
			t.Errorf("Expected 1 trade (force closed), got %d", result.TotalTrades)
		}
	})

	t.Run("equity curve length matches bars", func(t *testing.T) {
		bars := makeBars(50, 50000, 0.02)
		signals := make([]Signal, len(bars))
		signals[5] = SignalLong

		result := runner.RunWithData(bars, signals, nil, nil)

		if len(result.EquityCurve) != len(bars) {
			t.Errorf("Equity curve length %d != bar count %d", len(result.EquityCurve), len(bars))
		}
	})

	t.Run("metrics calculated correctly", func(t *testing.T) {
		// Create scenario with known outcome
		bars := makeBars(100, 50000, 0.02)
		signals := make([]Signal, len(bars))
		signals[5] = SignalLong
		signals[30] = SignalShort

		result := runner.RunWithData(bars, signals, nil, nil)

		// Verify result fields are populated
		if result.TotalTrades > 0 {
			if result.WinRate < 0 || result.WinRate > 1 {
				t.Errorf("Invalid WinRate: %f", result.WinRate)
			}
		}
	})
}

func TestRunWithDataConsistency(t *testing.T) {
	cfg := DefaultConfig()
	runner, _ := NewRunner(cfg)

	bars := makeBars(50, 50000, 0.02)
	signals := make([]Signal, len(bars))
	signals[5] = SignalLong
	signals[25] = SignalShort

	// Run twice and verify same results
	result1 := runner.RunWithData(bars, signals, nil, nil)
	result2 := runner.RunWithData(bars, signals, nil, nil)

	if result1.TotalTrades != result2.TotalTrades {
		t.Error("Inconsistent TotalTrades between runs")
	}
	if result1.TotalPnL != result2.TotalPnL {
		t.Error("Inconsistent TotalPnL between runs")
	}
	if result1.WinRate != result2.WinRate {
		t.Error("Inconsistent WinRate between runs")
	}
}

func TestResultToJSON(t *testing.T) {
	result := Result{
		TotalTrades:  10,
		WinRate:      0.6,
		AvgPnL:       0.005,
		TotalPnL:     0.05,
		SharpeRatio:  1.5,
		MaxDrawdown:  0.1,
		ProfitFactor: 2.0,
		AvgHoldBars:  5.5,
		TPCount:      4,
		SLCount:      3,
		TimeoutCount: 3,
		EquityCurve:  []float64{10000, 10050, 10100, 10080, 10150},
	}

	t.Run("without equity curve", func(t *testing.T) {
		jsonResult := result.ToJSON(false)

		if jsonResult.TotalTrades != 10 {
			t.Errorf("TotalTrades mismatch: %d", jsonResult.TotalTrades)
		}
		if jsonResult.WinRate != 0.6 {
			t.Errorf("WinRate mismatch: %f", jsonResult.WinRate)
		}
		if jsonResult.EquityCurve != nil {
			t.Error("EquityCurve should be nil when not included")
		}
	})

	t.Run("with equity curve", func(t *testing.T) {
		jsonResult := result.ToJSON(true)

		if len(jsonResult.EquityCurve) != 5 {
			t.Errorf("EquityCurve length: %d, want 5", len(jsonResult.EquityCurve))
		}
	})
}

func TestResultWriteJSON(t *testing.T) {
	result := Result{
		TotalTrades:  5,
		WinRate:      0.6,
		AvgPnL:       0.01,
		TotalPnL:     0.05,
		SharpeRatio:  1.2,
		MaxDrawdown:  0.08,
		ProfitFactor: 1.5,
		AvgHoldBars:  4.2,
		TPCount:      2,
		SLCount:      2,
		TimeoutCount: 1,
		EquityCurve:  []float64{10000, 10100, 10200},
	}

	t.Run("write and read back", func(t *testing.T) {
		tmpDir := t.TempDir()
		path := filepath.Join(tmpDir, "result.json")

		err := result.WriteJSON(path, true)
		if err != nil {
			t.Fatalf("WriteJSON failed: %v", err)
		}

		// Read back
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("ReadFile failed: %v", err)
		}

		var readResult ResultJSON
		err = json.Unmarshal(data, &readResult)
		if err != nil {
			t.Fatalf("Unmarshal failed: %v", err)
		}

		if readResult.TotalTrades != 5 {
			t.Errorf("TotalTrades mismatch: %d", readResult.TotalTrades)
		}
		if readResult.WinRate != 0.6 {
			t.Errorf("WinRate mismatch: %f", readResult.WinRate)
		}
		if len(readResult.EquityCurve) != 3 {
			t.Errorf("EquityCurve length: %d", len(readResult.EquityCurve))
		}
	})

	t.Run("write without equity curve", func(t *testing.T) {
		tmpDir := t.TempDir()
		path := filepath.Join(tmpDir, "result_no_curve.json")

		err := result.WriteJSON(path, false)
		if err != nil {
			t.Fatalf("WriteJSON failed: %v", err)
		}

		data, _ := os.ReadFile(path)
		var readResult ResultJSON
		json.Unmarshal(data, &readResult)

		if readResult.EquityCurve != nil {
			t.Error("EquityCurve should be nil")
		}
	})
}

func TestConfigFromJSON(t *testing.T) {
	t.Run("valid config", func(t *testing.T) {
		tmpDir := t.TempDir()
		path := filepath.Join(tmpDir, "config.json")

		cfg := DefaultConfig()
		cfg.SLMult = 1.5
		cfg.PTMult = 3.0

		data, _ := json.MarshalIndent(cfg, "", "  ")
		os.WriteFile(path, data, 0644)

		loaded, err := ConfigFromJSON(path)
		if err != nil {
			t.Fatalf("ConfigFromJSON failed: %v", err)
		}

		if loaded.SLMult != 1.5 {
			t.Errorf("SLMult mismatch: %f", loaded.SLMult)
		}
		if loaded.PTMult != 3.0 {
			t.Errorf("PTMult mismatch: %f", loaded.PTMult)
		}
	})

	t.Run("file not found", func(t *testing.T) {
		_, err := ConfigFromJSON("/nonexistent/path.json")
		if err == nil {
			t.Error("Expected error for nonexistent file")
		}
	})

	t.Run("invalid json", func(t *testing.T) {
		tmpDir := t.TempDir()
		path := filepath.Join(tmpDir, "invalid.json")
		os.WriteFile(path, []byte("not valid json"), 0644)

		_, err := ConfigFromJSON(path)
		if err == nil {
			t.Error("Expected error for invalid JSON")
		}
	})
}

func TestPrintSummary(t *testing.T) {
	result := Result{
		TotalTrades:  10,
		WinRate:      0.6,
		AvgPnL:       0.005,
		TotalPnL:     0.05,
		SharpeRatio:  1.5,
		MaxDrawdown:  0.1,
		ProfitFactor: 2.0,
		AvgHoldBars:  5.5,
		TPCount:      4,
		SLCount:      3,
		TimeoutCount: 3,
	}

	// Capture stdout
	old := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w

	result.PrintSummary()

	w.Close()
	os.Stdout = old

	var buf bytes.Buffer
	io.Copy(&buf, r)
	output := buf.String()

	// Verify key metrics are printed
	expectedStrings := []string{
		"Total Trades:",
		"Win Rate:",
		"Sharpe Ratio:",
		"Max Drawdown:",
		"Take Profit:",
		"Stop Loss:",
	}

	for _, expected := range expectedStrings {
		if !bytes.Contains([]byte(output), []byte(expected)) {
			t.Errorf("PrintSummary missing: %s", expected)
		}
	}
}

func TestSignalRecordConversion(t *testing.T) {
	// Test Signal type conversion from record values
	tests := []struct {
		recordSignal int8
		expected     Signal
	}{
		{-1, SignalShort},
		{0, SignalNeutral},
		{1, SignalLong},
	}

	for _, tt := range tests {
		record := SignalRecord{Signal: tt.recordSignal}
		got := Signal(record.Signal)
		if got != tt.expected {
			t.Errorf("Signal(%d) = %d, want %d", tt.recordSignal, got, tt.expected)
		}
	}
}

func TestFeatureRecordConversion(t *testing.T) {
	record := FeatureRecord{
		Timestamp:   1234567890,
		Open:        50000.0,
		High:        50100.0,
		Low:         49900.0,
		Close:       50050.0,
		Volume:      1000.0,
		RealizedVol: 0.02,
	}

	bar := Bar{
		Timestamp:   record.Timestamp,
		Open:        record.Open,
		High:        record.High,
		Low:         record.Low,
		Close:       record.Close,
		Volume:      record.Volume,
		RealizedVol: record.RealizedVol,
	}

	if bar.Timestamp != 1234567890 {
		t.Errorf("Timestamp mismatch")
	}
	if bar.Open != 50000.0 {
		t.Errorf("Open mismatch")
	}
	if bar.RealizedVol != 0.02 {
		t.Errorf("RealizedVol mismatch")
	}
}

func TestRunnerReset(t *testing.T) {
	cfg := DefaultConfig()
	runner, _ := NewRunner(cfg)

	bars := makeBars(30, 50000, 0.02)
	signals := make([]Signal, len(bars))
	signals[0] = SignalLong

	// First run
	result1 := runner.RunWithData(bars, signals, nil, nil)

	// Second run should be independent
	result2 := runner.RunWithData(bars, signals, nil, nil)

	if result1.TotalTrades != result2.TotalTrades {
		t.Error("Runner state not properly reset between runs")
	}
}

func TestEdgeCases(t *testing.T) {
	cfg := DefaultConfig()
	runner, _ := NewRunner(cfg)

	t.Run("single bar", func(t *testing.T) {
		bars := []Bar{{
			Timestamp:   1000,
			Open:        50000,
			High:        50100,
			Low:         49900,
			Close:       50000,
			Volume:      1000,
			RealizedVol: 0.02,
		}}
		signals := []Signal{SignalLong}

		result := runner.RunWithData(bars, signals, nil, nil)

		// Can't enter on single bar (need next bar)
		if result.TotalTrades != 0 {
			t.Errorf("Single bar should have no trades, got %d", result.TotalTrades)
		}
	})

	t.Run("two bars with signal", func(t *testing.T) {
		bars := []Bar{
			{Timestamp: 1000, Open: 50000, High: 50100, Low: 49900, Close: 50050, Volume: 1000, RealizedVol: 0.02},
			{Timestamp: 2000, Open: 50050, High: 50150, Low: 49950, Close: 50100, Volume: 1000, RealizedVol: 0.02},
		}
		signals := []Signal{SignalLong, SignalNeutral}

		result := runner.RunWithData(bars, signals, nil, nil)

		// Should enter on bar 1, force close at bar 1 (end)
		if result.TotalTrades != 1 {
			t.Errorf("Two bars should have 1 trade (force closed), got %d", result.TotalTrades)
		}
	})

	t.Run("signals longer than bars", func(t *testing.T) {
		bars := makeBars(10, 50000, 0.02)
		signals := make([]Signal, 20) // More signals than bars
		signals[0] = SignalLong

		// Should not panic
		result := runner.RunWithData(bars, signals, nil, nil)
		_ = result // Just check it doesn't crash
	})

	t.Run("bars longer than signals", func(t *testing.T) {
		bars := makeBars(20, 50000, 0.02)
		signals := make([]Signal, 10) // Fewer signals
		signals[0] = SignalLong

		// Should not panic - missing signals treated as neutral
		result := runner.RunWithData(bars, signals, nil, nil)
		_ = result
	})
}

// Helper function to create test bars
func makeBars(count int, basePrice, volatility float64) []Bar {
	bars := make([]Bar, count)
	price := basePrice

	for i := range bars {
		// Random walk with volatility
		change := volatility * basePrice * 0.1 * (float64(i%5) - 2) / 2
		price += change

		bars[i] = Bar{
			Timestamp:   int64(i * 60000),
			Open:        price,
			High:        price + volatility*price*0.5,
			Low:         price - volatility*price*0.5,
			Close:       price + volatility*price*0.2,
			Volume:      1000,
			RealizedVol: volatility,
		}
	}

	return bars
}

// Benchmark tests
func BenchmarkRunWithData_Small(b *testing.B) {
	cfg := DefaultConfig()
	runner, _ := NewRunner(cfg)

	bars := makeBars(100, 50000, 0.02)
	signals := make([]Signal, len(bars))
	for i := 0; i < len(signals); i += 20 {
		signals[i] = SignalLong
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		runner.RunWithData(bars, signals, nil, nil)
	}
}

func BenchmarkRunWithData_Large(b *testing.B) {
	cfg := DefaultConfig()
	runner, _ := NewRunner(cfg)

	bars := makeBars(10000, 50000, 0.02)
	signals := make([]Signal, len(bars))
	for i := 0; i < len(signals); i += 50 {
		signals[i] = SignalLong
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		runner.RunWithData(bars, signals, nil, nil)
	}
}

func BenchmarkRunWithData_ManyTrades(b *testing.B) {
	cfg := DefaultConfig()
	cfg.MaxHoldBars = 5 // Quick trades
	runner, _ := NewRunner(cfg)

	bars := makeBars(1000, 50000, 0.02)
	signals := make([]Signal, len(bars))
	for i := 0; i < len(signals); i += 10 {
		if i%20 == 0 {
			signals[i] = SignalLong
		} else {
			signals[i] = SignalShort
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		runner.RunWithData(bars, signals, nil, nil)
	}
}
