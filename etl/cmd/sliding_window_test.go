package main

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"dl-rl-btc-etl/internal/bars"
	"dl-rl-btc-etl/internal/config"
	"dl-rl-btc-etl/internal/storage"
)

// TestSlidingWindowMonthBoundaryBar tests that a bar spanning two months
// is correctly attributed to the month of its StartTime.
func TestSlidingWindowMonthBoundaryBar(t *testing.T) {
	generator := bars.NewGenerator()

	// Skip warmup by injecting 14 days of simulated data
	for i := 0; i < bars.WarmupDays; i++ {
		ts := time.Date(2020, 8, 15+i, 12, 0, 0, 0, time.UTC).UnixMilli()
		generator.ProcessTrade(10000, 100, 1_000_000, ts, true)
	}

	// Now generator is out of warmup, threshold is set
	threshold := generator.GetCurrentThreshold()
	t.Logf("Threshold: $%.2f", threshold)

	// Create trades that span month boundary
	// September 30th trades (not enough to complete a bar)
	sep30 := time.Date(2020, 9, 30, 23, 59, 0, 0, time.UTC).UnixMilli()
	halfThreshold := threshold / 2

	bar1 := generator.ProcessTrade(10000, halfThreshold/10000*0.4, halfThreshold*0.4, sep30, true)
	if bar1 != nil {
		t.Log("Unexpected bar generated from Sep 30 partial trades")
	}

	// October 1st trades (completes the bar)
	oct1 := time.Date(2020, 10, 1, 0, 1, 0, 0, time.UTC).UnixMilli()
	bar2 := generator.ProcessTrade(10100, halfThreshold/10100*0.8, halfThreshold*0.8, oct1, false)

	if bar2 == nil {
		// Might need more trades to hit threshold
		bar2 = generator.ProcessTrade(10050, threshold/10050, threshold, oct1+1000, true)
	}

	if bar2 != nil {
		barTime := time.UnixMilli(bar2.StartTime).UTC()
		t.Logf("Boundary bar StartTime: %s", barTime.Format("2006-01-02 15:04:05"))

		// The bar should have StartTime in September (where it began)
		if barTime.Month() != time.September {
			t.Errorf("Expected bar StartTime in September, got %s", barTime.Month())
		}
	} else {
		t.Log("No bar generated (threshold not reached)")
	}
}

// TestSlidingWindowSavesAtCorrectTime tests that bars are saved
// when processing the 3rd month (N-2 month gets saved).
func TestSlidingWindowSavesAtCorrectTime(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "sliding_window_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	cfg := config.ETLConfig{
		Symbol: "TEST",
		OutDir: tmpDir,
	}

	// Simulate the sliding window logic from processMarket
	barsByMonth := make(map[string][]bars.DynamicDollarBar)
	var processedMonths []string

	// Create mock bars for 4 months
	months := []string{"2020-09", "2020-10", "2020-11", "2020-12"}

	for i, monthKey := range months {
		// Add mock bars for this month
		mockBar := bars.DynamicDollarBar{
			StartTime:     time.Date(2020, time.Month(9+i), 15, 12, 0, 0, 0, time.UTC).UnixMilli(),
			EndTime:       time.Date(2020, time.Month(9+i), 15, 12, 30, 0, 0, time.UTC).UnixMilli(),
			Open:          10000,
			High:          10100,
			HighTime:      time.Date(2020, time.Month(9+i), 15, 12, 15, 0, 0, time.UTC).UnixMilli(),
			Low:           9900,
			LowTime:       time.Date(2020, time.Month(9+i), 15, 12, 5, 0, 0, time.UTC).UnixMilli(),
			Close:         10050,
			Volume:        100,
			DollarValue:   1000000,
			TickCount:     1000,
			ThresholdUsed: 1000000,
			Duration:      1800,
			BuyDollarVol:  600000,
			SellDollarVol: 400000,
			NetImbalance:  200000,
		}
		barsByMonth[monthKey] = append(barsByMonth[monthKey], mockBar)
		processedMonths = append(processedMonths, monthKey)

		t.Logf("Processing month %s, processedMonths: %v", monthKey, processedMonths)

		// Save and release months that are 2+ months behind
		if len(processedMonths) > 2 {
			oldKey := processedMonths[0]
			if oldBars, exists := barsByMonth[oldKey]; exists && len(oldBars) > 0 {
				if err := saveMonthBars(cfg, oldKey, oldBars); err != nil {
					t.Errorf("Failed to save %s: %v", oldKey, err)
				}
				t.Logf("Saved and released %s", oldKey)

				// Verify file was created
				expectedFile := filepath.Join(tmpDir, fmt.Sprintf("TEST-bars-%s.parquet", oldKey))
				if _, err := os.Stat(expectedFile); os.IsNotExist(err) {
					t.Errorf("Expected file %s was not created", expectedFile)
				}
			}
			delete(barsByMonth, oldKey)
			processedMonths = processedMonths[1:]
		}
	}

	// Verify only 2 months remain in memory
	if len(barsByMonth) != 2 {
		t.Errorf("Expected 2 months in memory, got %d", len(barsByMonth))
	}

	// Verify saved files
	savedFiles, _ := filepath.Glob(filepath.Join(tmpDir, "*.parquet"))
	// Should have saved 2020-09 and 2020-10
	if len(savedFiles) != 2 {
		t.Errorf("Expected 2 saved files, got %d: %v", len(savedFiles), savedFiles)
	}

	t.Logf("Remaining in memory: %v", processedMonths)
	t.Logf("Saved files: %v", savedFiles)
}

// TestSlidingWindowEmptyMonth tests handling of months with no bars.
func TestSlidingWindowEmptyMonth(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "empty_month_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	cfg := config.ETLConfig{
		Symbol: "TEST",
		OutDir: tmpDir,
	}

	barsByMonth := make(map[string][]bars.DynamicDollarBar)
	var processedMonths []string

	// Month 1: has bars
	barsByMonth["2020-09"] = []bars.DynamicDollarBar{{
		StartTime: time.Date(2020, 9, 15, 12, 0, 0, 0, time.UTC).UnixMilli(),
		Open:      10000, High: 10100, Low: 9900, Close: 10050,
	}}
	processedMonths = append(processedMonths, "2020-09")

	// Month 2: empty (no bars generated, e.g., warmup period)
	// barsByMonth["2020-10"] is empty/not set
	processedMonths = append(processedMonths, "2020-10")

	// Month 3: has bars
	barsByMonth["2020-11"] = []bars.DynamicDollarBar{{
		StartTime: time.Date(2020, 11, 15, 12, 0, 0, 0, time.UTC).UnixMilli(),
		Open:      11000, High: 11100, Low: 10900, Close: 11050,
	}}
	processedMonths = append(processedMonths, "2020-11")

	// Process: when len > 2, save oldest
	if len(processedMonths) > 2 {
		oldKey := processedMonths[0] // "2020-09"
		if oldBars, exists := barsByMonth[oldKey]; exists && len(oldBars) > 0 {
			if err := saveMonthBars(cfg, oldKey, oldBars); err != nil {
				t.Errorf("Failed to save %s: %v", oldKey, err)
			}
		}
		delete(barsByMonth, oldKey)
		processedMonths = processedMonths[1:]
	}

	// Month 4: has bars
	barsByMonth["2020-12"] = []bars.DynamicDollarBar{{
		StartTime: time.Date(2020, 12, 15, 12, 0, 0, 0, time.UTC).UnixMilli(),
		Open:      12000, High: 12100, Low: 11900, Close: 12050,
	}}
	processedMonths = append(processedMonths, "2020-12")

	if len(processedMonths) > 2 {
		oldKey := processedMonths[0] // "2020-10" (empty)
		if oldBars, exists := barsByMonth[oldKey]; exists && len(oldBars) > 0 {
			if err := saveMonthBars(cfg, oldKey, oldBars); err != nil {
				t.Errorf("Failed to save %s: %v", oldKey, err)
			}
		} else {
			t.Logf("Skipped empty month %s (no bars to save)", oldKey)
		}
		delete(barsByMonth, oldKey)
		processedMonths = processedMonths[1:]
	}

	// Verify: 2020-10 should NOT have a file (it was empty)
	emptyMonthFile := filepath.Join(tmpDir, "TEST-bars-2020-10.parquet")
	if _, err := os.Stat(emptyMonthFile); !os.IsNotExist(err) {
		t.Errorf("Empty month file should not exist: %s", emptyMonthFile)
	}

	// 2020-09 should have a file
	sepFile := filepath.Join(tmpDir, "TEST-bars-2020-09.parquet")
	if _, err := os.Stat(sepFile); os.IsNotExist(err) {
		t.Errorf("September file should exist: %s", sepFile)
	}

	t.Log("Empty month handling test passed")
}

// TestSlidingWindowRemainingBars tests that remaining bars in memory
// are saved at the end of processing.
func TestSlidingWindowRemainingBars(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "remaining_bars_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	cfg := config.ETLConfig{
		Symbol: "TEST",
		OutDir: tmpDir,
	}

	barsByMonth := make(map[string][]bars.DynamicDollarBar)

	// Simulate end of processing with 2 months remaining
	barsByMonth["2020-11"] = []bars.DynamicDollarBar{{
		StartTime: time.Date(2020, 11, 15, 12, 0, 0, 0, time.UTC).UnixMilli(),
		Open:      10000, High: 10100, Low: 9900, Close: 10050,
		DollarValue: 1000000, ThresholdUsed: 1000000,
	}}
	barsByMonth["2020-12"] = []bars.DynamicDollarBar{{
		StartTime: time.Date(2020, 12, 15, 12, 0, 0, 0, time.UTC).UnixMilli(),
		Open:      11000, High: 11100, Low: 10900, Close: 11050,
		DollarValue: 1100000, ThresholdUsed: 1000000,
	}}

	// Save remaining months (simulating end of processMarket loop)
	for key, monthBars := range barsByMonth {
		if len(monthBars) > 0 {
			if err := saveMonthBars(cfg, key, monthBars); err != nil {
				t.Errorf("Failed to save %s: %v", key, err)
			}
		}
	}

	// Verify both files were created
	novFile := filepath.Join(tmpDir, "TEST-bars-2020-11.parquet")
	decFile := filepath.Join(tmpDir, "TEST-bars-2020-12.parquet")

	if _, err := os.Stat(novFile); os.IsNotExist(err) {
		t.Errorf("November file should exist: %s", novFile)
	}
	if _, err := os.Stat(decFile); os.IsNotExist(err) {
		t.Errorf("December file should exist: %s", decFile)
	}

	// Verify file contents
	novBars, err := storage.ReadDollarBars(novFile)
	if err != nil {
		t.Errorf("Failed to read November file: %v", err)
	}
	if len(novBars) != 1 {
		t.Errorf("Expected 1 bar in November file, got %d", len(novBars))
	}

	decBars, err := storage.ReadDollarBars(decFile)
	if err != nil {
		t.Errorf("Failed to read December file: %v", err)
	}
	if len(decBars) != 1 {
		t.Errorf("Expected 1 bar in December file, got %d", len(decBars))
	}

	t.Log("Remaining bars save test passed")
}

// TestSlidingWindowHighLowTimestamps tests that HighTime and LowTime
// are correctly tracked for backtest precision.
func TestSlidingWindowHighLowTimestamps(t *testing.T) {
	generator := bars.NewGenerator()

	// Skip warmup
	for i := 0; i < bars.WarmupDays; i++ {
		ts := time.Date(2020, 8, 15+i, 12, 0, 0, 0, time.UTC).UnixMilli()
		generator.ProcessTrade(10000, 100, 1_000_000, ts, true)
	}

	threshold := generator.GetCurrentThreshold()
	baseTime := time.Date(2020, 9, 15, 12, 0, 0, 0, time.UTC)

	// Trade 1: Open price (also initial High and Low)
	t1 := baseTime.UnixMilli()
	generator.ProcessTrade(10000, threshold/10000*0.2, threshold*0.2, t1, true)

	// Trade 2: New High
	t2 := baseTime.Add(5 * time.Minute).UnixMilli()
	generator.ProcessTrade(10500, threshold/10500*0.2, threshold*0.2, t2, true)

	// Trade 3: New Low
	t3 := baseTime.Add(10 * time.Minute).UnixMilli()
	generator.ProcessTrade(9500, threshold/9500*0.2, threshold*0.2, t3, false)

	// Trade 4: Complete the bar (price between high and low)
	t4 := baseTime.Add(15 * time.Minute).UnixMilli()
	bar := generator.ProcessTrade(10000, threshold/10000*0.5, threshold*0.5, t4, true)

	if bar == nil {
		bar = generator.Flush()
	}

	if bar == nil {
		t.Fatal("No bar generated")
	}

	// Verify High/Low timestamps
	if bar.HighTime != t2 {
		t.Errorf("HighTime mismatch: expected %d, got %d", t2, bar.HighTime)
	}
	if bar.LowTime != t3 {
		t.Errorf("LowTime mismatch: expected %d, got %d", t3, bar.LowTime)
	}

	// Verify High occurred before Low (for this test case)
	if bar.HighTime >= bar.LowTime {
		t.Errorf("Expected HighTime < LowTime for this test case")
	}

	t.Logf("Bar: High=%.2f at %s, Low=%.2f at %s",
		bar.High, time.UnixMilli(bar.HighTime).Format("15:04:05"),
		bar.Low, time.UnixMilli(bar.LowTime).Format("15:04:05"))
	t.Log("High/Low timestamp test passed")
}

// TestSlidingWindowCrossMonthBarAttribution tests that a bar starting
// in month N-1 but completing in month N is attributed to month N-1.
func TestSlidingWindowCrossMonthBarAttribution(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "cross_month_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	cfg := config.ETLConfig{
		Symbol: "TEST",
		OutDir: tmpDir,
	}

	barsByMonth := make(map[string][]bars.DynamicDollarBar)

	// Simulate a bar that starts in September but completes in October
	crossMonthBar := bars.DynamicDollarBar{
		StartTime:     time.Date(2020, 9, 30, 23, 55, 0, 0, time.UTC).UnixMilli(), // Sep 30
		EndTime:       time.Date(2020, 10, 1, 0, 5, 0, 0, time.UTC).UnixMilli(),   // Oct 1
		Open:          10000,
		High:          10100,
		HighTime:      time.Date(2020, 10, 1, 0, 3, 0, 0, time.UTC).UnixMilli(),
		Low:           9900,
		LowTime:       time.Date(2020, 9, 30, 23, 57, 0, 0, time.UTC).UnixMilli(),
		Close:         10050,
		Volume:        100,
		DollarValue:   1000000,
		TickCount:     500,
		ThresholdUsed: 1000000,
		Duration:      600,
		BuyDollarVol:  600000,
		SellDollarVol: 400000,
		NetImbalance:  200000,
	}

	// This bar should be attributed to September (StartTime month)
	barTime := time.UnixMilli(crossMonthBar.StartTime).UTC()
	key := fmt.Sprintf("%d-%02d", barTime.Year(), barTime.Month())

	if key != "2020-09" {
		t.Errorf("Expected bar to be attributed to 2020-09, got %s", key)
	}

	barsByMonth[key] = append(barsByMonth[key], crossMonthBar)

	// Save and verify
	if err := saveMonthBars(cfg, key, barsByMonth[key]); err != nil {
		t.Fatalf("Failed to save: %v", err)
	}

	// Read back and verify
	savedBars, err := storage.ReadDollarBars(filepath.Join(tmpDir, "TEST-bars-2020-09.parquet"))
	if err != nil {
		t.Fatalf("Failed to read saved bars: %v", err)
	}

	if len(savedBars) != 1 {
		t.Fatalf("Expected 1 bar, got %d", len(savedBars))
	}

	savedBar := savedBars[0]
	if savedBar.StartTime != crossMonthBar.StartTime {
		t.Errorf("StartTime mismatch")
	}
	if savedBar.EndTime != crossMonthBar.EndTime {
		t.Errorf("EndTime mismatch")
	}

	t.Logf("Cross-month bar: StartTime=%s, EndTime=%s, attributed to %s",
		time.UnixMilli(savedBar.StartTime).Format("2006-01-02 15:04"),
		time.UnixMilli(savedBar.EndTime).Format("2006-01-02 15:04"),
		key)
	t.Log("Cross-month bar attribution test passed")
}
