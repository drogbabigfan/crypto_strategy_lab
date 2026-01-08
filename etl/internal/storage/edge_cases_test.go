package storage

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"dl-rl-btc-etl/internal/bars"
)

// ============================================================================
// RISK: Empty Data Handling
// ============================================================================

func TestWriteEmptyBars(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "storage_edge_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	path := filepath.Join(tmpDir, "empty.parquet")

	// Empty slice
	emptyBars := []bars.DynamicDollarBar{}

	err = WriteDollarBars(path, emptyBars)
	if err != nil {
		t.Fatalf("Should handle empty bars: %v", err)
	}

	// Verify file exists but may be minimal
	fi, err := os.Stat(path)
	if err != nil {
		t.Fatalf("File should exist: %v", err)
	}

	t.Logf("Empty parquet file size: %d bytes", fi.Size())
}

func TestReadEmptyParquet(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "storage_edge_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	path := filepath.Join(tmpDir, "empty.parquet")

	// Write empty
	err = WriteDollarBars(path, []bars.DynamicDollarBar{})
	if err != nil {
		t.Fatalf("Write failed: %v", err)
	}

	// Read back
	readBars, err := ReadDollarBars(path)
	if err != nil {
		t.Fatalf("Read failed: %v", err)
	}

	if len(readBars) != 0 {
		t.Errorf("Expected 0 bars, got %d", len(readBars))
	}
}

// ============================================================================
// RISK: Special Float Values
// ============================================================================

func TestWriteBarsWithNaN(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "storage_edge_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	path := filepath.Join(tmpDir, "nan.parquet")

	// Bar with NaN values
	barsWithNaN := []bars.DynamicDollarBar{
		{
			StartTime:     1600000000000,
			EndTime:       1600000001000,
			Open:          math.NaN(),
			High:          50000,
			Low:           49000,
			Close:         49500,
			Volume:        1.0,
			DollarValue:   50000,
			TickCount:     10,
			ThresholdUsed: 50000,
			Duration:      1.0,
			BuyDollarVol:  25000,
			SellDollarVol: 25000,
			NetImbalance:  0,
		},
	}

	err = WriteDollarBars(path, barsWithNaN)
	// Document current behavior with NaN
	if err != nil {
		t.Logf("NaN handling on write: %v", err)
	} else {
		// Read back and check
		readBars, err := ReadDollarBars(path)
		if err != nil {
			t.Logf("NaN handling on read: %v", err)
		} else if len(readBars) > 0 {
			if math.IsNaN(readBars[0].Open) {
				t.Log("NaN is preserved in Parquet")
			} else {
				t.Logf("NaN converted to: %f", readBars[0].Open)
			}
		}
	}
}

func TestWriteBarsWithInfinity(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "storage_edge_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	path := filepath.Join(tmpDir, "inf.parquet")

	// Bar with Infinity values
	barsWithInf := []bars.DynamicDollarBar{
		{
			StartTime:     1600000000000,
			EndTime:       1600000001000,
			Open:          50000,
			High:          math.Inf(1), // +Inf
			Low:           math.Inf(-1), // -Inf
			Close:         49500,
			Volume:        1.0,
			DollarValue:   50000,
			TickCount:     10,
			ThresholdUsed: 50000,
			Duration:      1.0,
		},
	}

	err = WriteDollarBars(path, barsWithInf)
	if err != nil {
		t.Logf("Infinity handling on write: %v", err)
	} else {
		readBars, err := ReadDollarBars(path)
		if err != nil {
			t.Logf("Infinity handling on read: %v", err)
		} else if len(readBars) > 0 {
			t.Logf("Infinity values: High=%f, Low=%f",
				readBars[0].High, readBars[0].Low)
		}
	}
}

// ============================================================================
// RISK: Extreme Values
// ============================================================================

func TestWriteBarsWithExtremeValues(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "storage_edge_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	path := filepath.Join(tmpDir, "extreme.parquet")

	extremeBars := []bars.DynamicDollarBar{
		{
			StartTime:     0, // Unix epoch
			EndTime:       math.MaxInt64,
			Open:          0,
			High:          math.MaxFloat64,
			Low:           math.SmallestNonzeroFloat64,
			Close:         1e-308, // Very small
			Volume:        1e308,  // Very large
			DollarValue:   math.MaxFloat64,
			TickCount:     math.MaxInt64,
			ThresholdUsed: 0,
			Duration:      0.001, // MinDuration
		},
	}

	err = WriteDollarBars(path, extremeBars)
	if err != nil {
		t.Fatalf("Should handle extreme values: %v", err)
	}

	readBars, err := ReadDollarBars(path)
	if err != nil {
		t.Fatalf("Should read extreme values: %v", err)
	}

	if len(readBars) != 1 {
		t.Fatalf("Expected 1 bar, got %d", len(readBars))
	}

	// Verify values are preserved (within float precision)
	if readBars[0].TickCount != math.MaxInt64 {
		t.Errorf("TickCount not preserved: expected %d, got %d",
			int64(math.MaxInt64), readBars[0].TickCount)
	}
}

// ============================================================================
// RISK: Large Dataset
// ============================================================================

func TestWriteManyBars(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping large dataset test in short mode")
	}

	tmpDir, err := os.MkdirTemp("", "storage_edge_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	path := filepath.Join(tmpDir, "many.parquet")

	// Generate 100,000 bars
	numBars := 100_000
	manyBars := make([]bars.DynamicDollarBar, numBars)

	for i := 0; i < numBars; i++ {
		manyBars[i] = bars.DynamicDollarBar{
			StartTime:     int64(1600000000000 + i*1000),
			EndTime:       int64(1600000000000 + i*1000 + 500),
			Open:          50000 + float64(i%1000),
			High:          51000 + float64(i%1000),
			Low:           49000 + float64(i%1000),
			Close:         50500 + float64(i%1000),
			Volume:        float64(i%100) + 1,
			DollarValue:   50000 * (float64(i%100) + 1),
			TickCount:     int64(i%1000) + 1,
			ThresholdUsed: 1000000,
			Duration:      0.5,
			BuyDollarVol:  25000,
			SellDollarVol: 25000,
			NetImbalance:  0,
		}
	}

	err = WriteDollarBars(path, manyBars)
	if err != nil {
		t.Fatalf("Should handle many bars: %v", err)
	}

	fi, _ := os.Stat(path)
	t.Logf("100K bars parquet file size: %.2f MB", float64(fi.Size())/(1024*1024))

	// Read back and verify count
	readBars, err := ReadDollarBars(path)
	if err != nil {
		t.Fatalf("Should read many bars: %v", err)
	}

	if len(readBars) != numBars {
		t.Errorf("Expected %d bars, got %d", numBars, len(readBars))
	}
}

// ============================================================================
// RISK: File System Edge Cases
// ============================================================================

func TestWriteToReadOnlyDirectory(t *testing.T) {
	// This test may not work on all systems
	if os.Getuid() == 0 {
		t.Skip("Skipping as root user")
	}

	tmpDir, err := os.MkdirTemp("", "storage_edge_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Make directory read-only
	os.Chmod(tmpDir, 0444)
	defer os.Chmod(tmpDir, 0755) // Restore for cleanup

	path := filepath.Join(tmpDir, "readonly.parquet")

	testBars := []bars.DynamicDollarBar{
		{StartTime: 1600000000000, Open: 50000},
	}

	err = WriteDollarBars(path, testBars)
	if err == nil {
		t.Error("Should error when writing to read-only directory")
	}
}

func TestReadNonExistentFile(t *testing.T) {
	_, err := ReadDollarBars("/nonexistent/path/file.parquet")
	if err == nil {
		t.Error("Should error when reading non-existent file")
	}
}

func TestReadCorruptedFile(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "storage_edge_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	path := filepath.Join(tmpDir, "corrupted.parquet")

	// Write garbage data
	err = os.WriteFile(path, []byte("not a parquet file"), 0644)
	if err != nil {
		t.Fatalf("Failed to write corrupt file: %v", err)
	}

	_, err = ReadDollarBars(path)
	if err == nil {
		t.Error("Should error when reading corrupted file")
	}
}

// ============================================================================
// RISK: Zero Values
// ============================================================================

func TestWriteBarsWithAllZeros(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "storage_edge_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	path := filepath.Join(tmpDir, "zeros.parquet")

	// Bar with all zero values
	zeroBars := []bars.DynamicDollarBar{
		{}, // All fields default to zero
	}

	err = WriteDollarBars(path, zeroBars)
	if err != nil {
		t.Fatalf("Should handle zero bars: %v", err)
	}

	readBars, err := ReadDollarBars(path)
	if err != nil {
		t.Fatalf("Should read zero bars: %v", err)
	}

	if len(readBars) != 1 {
		t.Fatalf("Expected 1 bar, got %d", len(readBars))
	}

	// All values should be zero
	bar := readBars[0]
	if bar.StartTime != 0 || bar.Open != 0 || bar.Volume != 0 {
		t.Error("Zero values not preserved")
	}
}

// ============================================================================
// RISK: Negative Values
// ============================================================================

func TestWriteBarsWithNegativeValues(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "storage_edge_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	path := filepath.Join(tmpDir, "negative.parquet")

	// Bar with negative values (shouldn't happen but test handling)
	negativeBars := []bars.DynamicDollarBar{
		{
			StartTime:    -1,
			Open:         -50000,
			NetImbalance: -100000, // This is valid - can be negative
			Duration:     -1,      // Invalid but test storage
		},
	}

	err = WriteDollarBars(path, negativeBars)
	if err != nil {
		t.Logf("Negative value handling on write: %v", err)
		return
	}

	readBars, err := ReadDollarBars(path)
	if err != nil {
		t.Logf("Negative value handling on read: %v", err)
		return
	}

	if len(readBars) > 0 {
		t.Logf("Negative values preserved: StartTime=%d, Open=%f, Duration=%f",
			readBars[0].StartTime, readBars[0].Open, readBars[0].Duration)
	}
}
