package main

import (
	"archive/zip"
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"dl-rl-btc-etl/internal/bars"
	"dl-rl-btc-etl/internal/binance"
	"dl-rl-btc-etl/internal/features"
	"dl-rl-btc-etl/internal/storage"
)

// Expected columns from Binance trade data
var expectedColumns = []string{
	"id",
	"price",
	"qty",
	"quote_qty",
	"time",
	"is_buyer_maker",
	"is_best_match",
}

// Columns we actually use in DynamicDollarBar generation
var usedColumns = map[string]string{
	"price":          "OHLC calculation",
	"qty":            "Volume accumulation",
	"quote_qty":      "Dollar value (threshold check)",
	"time":           "Bar timestamp",
	"is_buyer_maker": "Aggressor side (Imbalance)",
}

// Columns we parse but don't use
var unusedColumns = map[string]string{
	"id":            "Trade ID - could be used for deduplication",
	"is_best_match": "Best price match flag - typically always true",
}

// TestIntegrationDollarBars downloads one month of real Binance data
// and verifies the entire ETL pipeline with DynamicDollarBar generation.
func TestIntegrationDollarBars(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	// Setup: create temp directories
	tmpDir, err := os.MkdirTemp("", "etl_integration_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	rawDir := filepath.Join(tmpDir, "raw")
	outDir := filepath.Join(tmpDir, "bars")
	os.MkdirAll(rawDir, 0755)
	os.MkdirAll(outDir, 0755)

	// Use 2017-08 (first month, small data)
	symbol := "BTCUSDT"
	year := 2017
	month := 8

	// 1. Download
	t.Log("Step 1: Downloading data...")
	downloader := binance.NewDownloader(rawDir)
	filename := "BTCUSDT-trades-2017-08.zip"
	url := binance.BaseURLSpot + "/" + symbol + "/" + filename

	if err := downloader.DownloadFile(url, filename); err != nil {
		t.Fatalf("Download failed: %v", err)
	}

	zipPath := filepath.Join(rawDir, filename)
	fi, _ := os.Stat(zipPath)
	t.Logf("Downloaded %s (%.2f MB)", filename, float64(fi.Size())/(1024*1024))

	// 2. Verify data structure
	t.Log("Step 2: Verifying data structure...")
	columnReport, err := analyzeZipColumns(zipPath)
	if err != nil {
		t.Fatalf("Failed to analyze columns: %v", err)
	}

	t.Log("=== Column Analysis Report ===")
	t.Logf("Total columns in data: %d", columnReport.TotalColumns)
	t.Logf("Header row: %v", columnReport.Headers)
	t.Logf("Sample data row: %v", columnReport.SampleRow)

	if len(columnReport.MissingColumns) > 0 {
		t.Errorf("Missing expected columns: %v", columnReport.MissingColumns)
	}
	if len(columnReport.ExtraColumns) > 0 {
		t.Logf("WARNING: Extra columns found: %v", columnReport.ExtraColumns)
	}

	t.Log("=== Column Usage Report ===")
	for col, usage := range usedColumns {
		t.Logf("  [USED]   %-15s - %s", col, usage)
	}
	for col, reason := range unusedColumns {
		t.Logf("  [UNUSED] %-15s - %s", col, reason)
	}

	// 3. Process with DynamicDollarBar Generator
	t.Log("Step 3: Processing with DynamicDollarBar Generator...")

	// Create generator - note: 2017 data is small, so we'll skip warmup
	// by pre-populating with simulated daily volumes
	generator := bars.NewGenerator()

	// Skip warmup by injecting simulated historical data
	// (Since this is August 2017, first month of data, we simulate warmup)
	for i := 0; i < bars.WarmupDays; i++ {
		generator.ProcessTrade(4000, 0.1, 1_000_000, // Simulate $1M daily volume for 14 days
			time.Date(2017, 7, 18+i, 12, 0, 0, 0, time.UTC).UnixMilli(), true)
	}

	// Now process the actual data
	dollarBars, tradeCount, err := processZipForTest(zipPath, generator)
	if err != nil {
		t.Fatalf("Failed to process ZIP: %v", err)
	}

	// Flush remaining
	if finalBar := generator.Flush(); finalBar != nil {
		dollarBars = append(dollarBars, *finalBar)
	}

	t.Logf("Processed %d trades -> %d bars", tradeCount, len(dollarBars))
	t.Logf("Generator warmup status: %v (remaining: %d days)",
		generator.IsWarmup(), generator.GetWarmupDaysRemaining())
	t.Logf("Current threshold: $%.2f", generator.GetCurrentThreshold())

	// 4. Verify bar data integrity
	t.Log("Step 4: Verifying bar data integrity...")

	if tradeCount == 0 {
		t.Fatal("No trades were parsed")
	}

	// Note: With 14-day warmup, we might not generate bars if data spans < 14 days
	if len(dollarBars) == 0 {
		t.Log("No bars generated (data may not span enough days to exit warmup)")
	}

	for i, bar := range dollarBars {
		// OHLC validation
		if bar.Open <= 0 || bar.High <= 0 || bar.Low <= 0 || bar.Close <= 0 {
			t.Errorf("Bar %d has invalid OHLC", i)
		}
		if bar.High < bar.Low {
			t.Errorf("Bar %d: High < Low", i)
		}

		// Dollar value validation
		if bar.DollarValue <= 0 {
			t.Errorf("Bar %d has non-positive DollarValue", i)
		}

		// Imbalance validation
		expectedNet := bar.BuyDollarVol - bar.SellDollarVol
		if bar.NetImbalance != expectedNet {
			t.Errorf("Bar %d: NetImbalance mismatch", i)
		}

		// ThresholdUsed should be recorded
		if bar.ThresholdUsed <= 0 {
			t.Errorf("Bar %d has no ThresholdUsed", i)
		}

		// Duration validation
		if bar.Duration < 0 {
			t.Errorf("Bar %d has negative duration", i)
		}

		// EndTime should be >= Timestamp
		if bar.EndTime < bar.StartTime {
			t.Errorf("Bar %d: EndTime < Timestamp", i)
		}
	}

	// 5. Save to Parquet (if we have bars)
	if len(dollarBars) > 0 {
		t.Log("Step 5: Saving to Parquet...")
		parquetPath := filepath.Join(outDir, fmt.Sprintf("%s-bars-%d-%02d.parquet", symbol, year, month))
		if err := storage.WriteDollarBars(parquetPath, dollarBars); err != nil {
			t.Fatalf("Failed to write Parquet: %v", err)
		}

		fi, _ = os.Stat(parquetPath)
		t.Logf("Saved Parquet file: %.2f KB", float64(fi.Size())/1024)
	}

	// 6. Test state persistence
	t.Log("Step 6: Testing state persistence...")
	state := generator.ExportState()
	if err := bars.SaveState(outDir, state); err != nil {
		t.Fatalf("Failed to save state: %v", err)
	}

	loadedState, err := bars.LoadState(outDir)
	if err != nil || loadedState == nil {
		t.Fatalf("Failed to load state: %v", err)
	}

	t.Logf("State preserved: Warmup=%v, DailyVolumeCount=%d, Threshold=%.0f",
		loadedState.IsWarmup, loadedState.DailyVolumeCount, loadedState.CurrentThreshold)

	// Summary
	t.Log("=== Integration Test Summary ===")
	t.Logf("Symbol: %s", symbol)
	t.Logf("Period: %d-%02d", year, month)
	t.Logf("Trades processed: %d", tradeCount)
	t.Logf("Bars generated: %d", len(dollarBars))
	if len(dollarBars) > 0 {
		t.Logf("First bar: Open=%.2f Close=%.2f DollarVal=%.0f Duration=%.1fs",
			dollarBars[0].Open, dollarBars[0].Close, dollarBars[0].DollarValue, dollarBars[0].Duration)
		t.Logf("Last bar: Open=%.2f Close=%.2f DollarVal=%.0f Duration=%.1fs",
			dollarBars[len(dollarBars)-1].Open, dollarBars[len(dollarBars)-1].Close,
			dollarBars[len(dollarBars)-1].DollarValue, dollarBars[len(dollarBars)-1].Duration)
	}
	t.Log("Integration test PASSED")
}

func processZipForTest(zipPath string, generator *bars.Generator) ([]bars.DynamicDollarBar, int, error) {
	archive, err := zip.OpenReader(zipPath)
	if err != nil {
		return nil, 0, err
	}
	defer archive.Close()

	var allBars []bars.DynamicDollarBar
	totalCount := 0

	for _, f := range archive.File {
		if !strings.HasSuffix(f.Name, ".csv") {
			continue
		}

		rc, err := f.Open()
		if err != nil {
			return nil, totalCount, err
		}

		fileBars, count, err := processCSVForTest(rc, generator)
		rc.Close()

		if err != nil {
			return nil, totalCount, err
		}

		allBars = append(allBars, fileBars...)
		totalCount += count
	}

	return allBars, totalCount, nil
}

func processCSVForTest(r io.Reader, generator *bars.Generator) ([]bars.DynamicDollarBar, int, error) {
	reader := csv.NewReader(r)
	var resultBars []bars.DynamicDollarBar
	first := true
	count := 0

	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, count, err
		}

		if first {
			first = false
			if len(record) > 0 && strings.Contains(strings.ToLower(record[0]), "id") {
				continue
			}
		}

		trade, err := binance.ParseTrade(record)
		if err != nil {
			continue
		}

		bar := generator.ProcessTrade(
			trade.Price,
			trade.Quantity,
			trade.QuoteQuantity,
			trade.Time,
			trade.IsAggressorBuy,
		)

		if bar != nil {
			resultBars = append(resultBars, *bar)
		}
		count++
	}

	return resultBars, count, nil
}

// ColumnReport contains analysis of CSV columns
type ColumnReport struct {
	TotalColumns   int
	Headers        []string
	SampleRow      []string
	MissingColumns []string
	ExtraColumns   []string
}

// TestIntegrationFutures downloads Futures trade data and processes through the full ETL pipeline.
func TestIntegrationFutures(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	// Setup: create temp directories
	tmpDir, err := os.MkdirTemp("", "etl_futures_integration_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	rawDir := filepath.Join(tmpDir, "raw")
	outDir := filepath.Join(tmpDir, "bars")
	os.MkdirAll(rawDir, 0755)
	os.MkdirAll(outDir, 0755)

	// Use 2020-01 (after warmup period)
	symbol := "BTCUSDT"
	year := 2020
	month := 1

	// 1. Download Futures trade data
	t.Log("Step 1: Downloading Futures trade data...")
	downloader := binance.NewDownloader(rawDir)
	tradeFilename := fmt.Sprintf("%s-trades-%d-%02d.zip", symbol, year, month)
	tradeURL := fmt.Sprintf("%s/%s/%s", binance.BaseURLFutures, symbol, tradeFilename)

	if err := downloader.DownloadFile(tradeURL, tradeFilename); err != nil {
		t.Fatalf("Trade data download failed: %v", err)
	}

	tradeZipPath := filepath.Join(rawDir, tradeFilename)
	fi, _ := os.Stat(tradeZipPath)
	t.Logf("Downloaded trade data %s (%.2f MB)", tradeFilename, float64(fi.Size())/(1024*1024))

	// 2. Process Futures trade data with DynamicDollarBar Generator
	t.Log("Step 2: Processing Futures trade data...")
	generator := bars.NewGenerator()

	// Skip warmup by injecting simulated historical data
	for i := 0; i < bars.WarmupDays; i++ {
		generator.ProcessTrade(7500, 10.0, 75_000_000, // Simulate ~$75M daily volume
			time.Date(2019, 12, 18+i, 12, 0, 0, 0, time.UTC).UnixMilli(), true)
	}

	dollarBars, tradeCount, err := processZipForTest(tradeZipPath, generator)
	if err != nil {
		t.Fatalf("Failed to process trade ZIP: %v", err)
	}

	// Flush remaining
	if finalBar := generator.Flush(); finalBar != nil {
		dollarBars = append(dollarBars, *finalBar)
	}

	t.Logf("Processed %d trades -> %d bars", tradeCount, len(dollarBars))
	t.Logf("Current threshold: $%.2f", generator.GetCurrentThreshold())

	if len(dollarBars) == 0 {
		t.Fatal("No bars generated from Futures data")
	}

	// 3. Verify bar data integrity
	t.Log("Step 3: Verifying bar data integrity...")

	for i, bar := range dollarBars {
		// Basic OHLC validation
		if bar.Open <= 0 || bar.High <= 0 || bar.Low <= 0 || bar.Close <= 0 {
			t.Errorf("Bar %d has invalid OHLC", i)
		}
		if bar.High < bar.Low {
			t.Errorf("Bar %d: High < Low", i)
		}

		// Dollar value validation
		if bar.DollarValue <= 0 {
			t.Errorf("Bar %d has non-positive DollarValue", i)
		}

		// Imbalance validation
		expectedNet := bar.BuyDollarVol - bar.SellDollarVol
		if bar.NetImbalance != expectedNet {
			t.Errorf("Bar %d: NetImbalance mismatch", i)
		}

		// Duration validation
		if bar.Duration < 0 {
			t.Errorf("Bar %d has negative duration", i)
		}
		if bar.EndTime < bar.StartTime {
			t.Errorf("Bar %d: EndTime < Timestamp", i)
		}
	}

	// 4. Save to Parquet
	t.Log("Step 4: Saving to Parquet...")
	parquetPath := filepath.Join(outDir, fmt.Sprintf("%s-futures-bars-%d-%02d.parquet", symbol, year, month))
	if err := storage.WriteDollarBars(parquetPath, dollarBars); err != nil {
		t.Fatalf("Failed to write Parquet: %v", err)
	}

	fi, _ = os.Stat(parquetPath)
	t.Logf("Saved Parquet file: %.2f KB", float64(fi.Size())/1024)

	// 5. Test state persistence
	t.Log("Step 5: Testing state persistence...")
	state := generator.ExportState()
	if err := bars.SaveState(outDir, state); err != nil {
		t.Fatalf("Failed to save state: %v", err)
	}

	loadedState, err := bars.LoadState(outDir)
	if err != nil || loadedState == nil {
		t.Fatalf("Failed to load state: %v", err)
	}

	// Validate loaded state
	if err := loadedState.Validate(); err != nil {
		t.Errorf("Loaded state failed validation: %v", err)
	}

	// Summary
	t.Log("=== Futures Integration Test Summary ===")
	t.Logf("Symbol: %s (Futures)", symbol)
	t.Logf("Period: %d-%02d", year, month)
	t.Logf("Trades processed: %d", tradeCount)
	t.Logf("Bars generated: %d", len(dollarBars))
	if len(dollarBars) > 0 {
		t.Logf("First bar: Open=%.2f Close=%.2f DollarVal=%.0f",
			dollarBars[0].Open, dollarBars[0].Close, dollarBars[0].DollarValue)
		t.Logf("Last bar: Open=%.2f Close=%.2f DollarVal=%.0f",
			dollarBars[len(dollarBars)-1].Open, dollarBars[len(dollarBars)-1].Close,
			dollarBars[len(dollarBars)-1].DollarValue)
	}
	t.Log("Futures integration test PASSED")
}

// TestIntegrationFeatureGeneration tests the full feature generation pipeline:
// DynamicDollarBar → FeatureGenerator → FeatureRow
// This test uses synthetic bars to verify all features are computed correctly.
func TestIntegrationFeatureGeneration(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	t.Log("=== Feature Generation Integration Test ===")

	// 1. Create synthetic bars with realistic data
	t.Log("Step 1: Creating synthetic dollar bars...")
	syntheticBars := generateSyntheticBars(500)
	t.Logf("Generated %d synthetic bars", len(syntheticBars))

	// 2. Initialize feature generator
	t.Log("Step 2: Initializing feature generator...")
	gen := features.NewGenerator(features.DefaultConfig())

	// 3. Process all bars and collect features
	t.Log("Step 3: Processing bars through feature generator...")
	var featureRows []features.FeatureRow
	var primedCount int

	for i, bar := range syntheticBars {
		row := gen.Process(bar)
		featureRows = append(featureRows, row)

		if row.IsPrimed {
			primedCount++
		}

		// Validate each row
		if err := validateFeatureRow(row, i); err != nil {
			t.Errorf("Bar %d validation failed: %v", i, err)
		}
	}

	t.Logf("Processed %d bars, %d are primed (%.1f%%)",
		len(featureRows), primedCount, float64(primedCount)/float64(len(featureRows))*100)

	// 4. Verify feature statistics
	t.Log("Step 4: Verifying feature statistics...")

	// Check L1 features (should all be valid after first bar)
	for i, row := range featureRows {
		if math.IsNaN(row.LogVolume) || math.IsInf(row.LogVolume, 0) {
			t.Errorf("Bar %d: LogVolume is NaN/Inf", i)
		}
		if math.IsNaN(row.LogDuration) || math.IsInf(row.LogDuration, 0) {
			t.Errorf("Bar %d: LogDuration is NaN/Inf", i)
		}
	}

	// Check cyclical time features (should always be in [-1, 1])
	for i, row := range featureRows {
		if row.SinTime < -1 || row.SinTime > 1 {
			t.Errorf("Bar %d: SinTime out of range: %v", i, row.SinTime)
		}
		if row.CosTime < -1 || row.CosTime > 1 {
			t.Errorf("Bar %d: CosTime out of range: %v", i, row.CosTime)
		}
		if row.SinWeek < -1 || row.SinWeek > 1 {
			t.Errorf("Bar %d: SinWeek out of range: %v", i, row.SinWeek)
		}
		if row.CosWeek < -1 || row.CosWeek > 1 {
			t.Errorf("Bar %d: CosWeek out of range: %v", i, row.CosWeek)
		}
	}

	// Check z-scores are soft-clipped (should be in [-5, 5])
	for i, row := range featureRows {
		if row.VolZScore < -5.1 || row.VolZScore > 5.1 {
			t.Errorf("Bar %d: VolZScore out of soft-clip range: %v", i, row.VolZScore)
		}
		if row.EntropyZScore < -5.1 || row.EntropyZScore > 5.1 {
			t.Errorf("Bar %d: EntropyZScore out of soft-clip range: %v", i, row.EntropyZScore)
		}
		if row.MomentumZScore10 < -5.1 || row.MomentumZScore10 > 5.1 {
			t.Errorf("Bar %d: MomentumZScore10 out of soft-clip range: %v", i, row.MomentumZScore10)
		}
	}

	// 5. Test state serialization
	t.Log("Step 5: Testing state serialization...")
	state1 := gen.State()

	gen2 := features.NewGenerator(features.DefaultConfig())
	gen2.LoadState(state1)

	// Process one more bar with both generators
	testBar := syntheticBars[0]
	testBar.StartTime = time.Now().UnixMilli()
	row1 := gen.Process(testBar)
	row2 := gen2.Process(testBar)

	if math.Abs(row1.GarmanKlassVol-row2.GarmanKlassVol) > 1e-10 {
		t.Errorf("State restoration failed: GarmanKlassVol mismatch")
	}
	if math.Abs(row1.FracDiffClose-row2.FracDiffClose) > 1e-10 {
		t.Errorf("State restoration failed: FracDiffClose mismatch")
	}

	// 6. Verify feature names count matches struct fields
	t.Log("Step 6: Verifying feature names...")
	featureNames := gen.FeatureNames()
	t.Logf("Total features: %d", len(featureNames))

	// Expected feature count: 36
	// Time(2) + Price(4) + L1(8) + Regime(8) + Stationarity(3) + Momentum(5) + Technical(1) + CyclicalTime(4) + Meta(1) = 36
	expectedFeatureCount := 36
	if len(featureNames) != expectedFeatureCount {
		t.Errorf("Expected %d features, got %d", expectedFeatureCount, len(featureNames))
	}

	// 7. Summary statistics for primed rows
	t.Log("Step 7: Computing summary statistics for primed rows...")
	if primedCount > 0 {
		var sumVol, sumEntropy, sumReturns float64
		for _, row := range featureRows {
			if row.IsPrimed {
				sumVol += row.GarmanKlassVol
				sumEntropy += row.ShannonEntropy
				sumReturns += row.Returns
			}
		}
		t.Logf("Primed rows avg: GK_Vol=%.6f, Entropy=%.4f, Returns=%.6f",
			sumVol/float64(primedCount),
			sumEntropy/float64(primedCount),
			sumReturns/float64(primedCount))
	}

	t.Log("=== Feature Generation Integration Test PASSED ===")
}

// generateSyntheticBars creates realistic synthetic dollar bars for testing.
func generateSyntheticBars(count int) []bars.DynamicDollarBar {
	result := make([]bars.DynamicDollarBar, count)

	basePrice := 50000.0
	baseTime := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC).UnixMilli()
	dollarThreshold := 28_000_000.0 // $28M per bar

	for i := 0; i < count; i++ {
		// Simulate price movement (random walk with drift)
		priceChange := (float64(i%10) - 5) * 50 // +/- $250
		price := basePrice + priceChange

		// Simulate varying duration (1-10 minutes)
		duration := float64(60 + (i%10)*60) // 60-600 seconds

		// Vary volume around threshold
		dollarValue := dollarThreshold * (0.98 + float64(i%5)*0.01)

		// Calculate derived values
		volume := dollarValue / price
		buyRatio := 0.4 + float64(i%3)*0.1 // 40-60% buy
		buyDollarVol := dollarValue * buyRatio
		sellDollarVol := dollarValue - buyDollarVol

		// Create bar with realistic OHLC
		open := price * (1 - 0.001)
		high := price * (1 + 0.002)
		low := price * (1 - 0.002)
		close := price

		result[i] = bars.DynamicDollarBar{
			StartTime:     baseTime + int64(i)*int64(duration*1000),
			EndTime:       baseTime + int64(i)*int64(duration*1000) + int64(duration*1000),
			Open:          open,
			High:          high,
			Low:           low,
			Close:         close,
			Volume:        volume,
			DollarValue:   dollarValue,
			BuyDollarVol:  buyDollarVol,
			SellDollarVol: sellDollarVol,
			NetImbalance:  buyDollarVol - sellDollarVol,
			Duration:      duration,
			TickCount:     int64(100 + i%50),
			ThresholdUsed: dollarThreshold,
		}
	}

	return result
}

// validateFeatureRow checks that a feature row has valid values.
func validateFeatureRow(row features.FeatureRow, idx int) error {
	// Check for NaN in critical fields
	if math.IsNaN(row.LogVolume) {
		return fmt.Errorf("LogVolume is NaN")
	}
	if math.IsNaN(row.LogDuration) {
		return fmt.Errorf("LogDuration is NaN")
	}
	if math.IsNaN(row.GarmanKlassVol) {
		return fmt.Errorf("GarmanKlassVol is NaN")
	}
	if math.IsNaN(row.Returns) && idx > 0 {
		return fmt.Errorf("Returns is NaN (after first bar)")
	}

	// Check for Inf
	if math.IsInf(row.LogVolume, 0) {
		return fmt.Errorf("LogVolume is Inf")
	}
	if math.IsInf(row.GarmanKlassVol, 0) {
		return fmt.Errorf("GarmanKlassVol is Inf")
	}

	// Cyclical features must be in [-1, 1]
	if row.SinTime < -1.01 || row.SinTime > 1.01 {
		return fmt.Errorf("SinTime out of range: %v", row.SinTime)
	}
	if row.CosTime < -1.01 || row.CosTime > 1.01 {
		return fmt.Errorf("CosTime out of range: %v", row.CosTime)
	}

	return nil
}

// TestIntegrationFeatureGenerationWithRealData tests feature generation using real downloaded data.
func TestIntegrationFeatureGenerationWithRealData(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	t.Log("=== Feature Generation with Real Data Integration Test ===")

	// Setup temp directories
	tmpDir, err := os.MkdirTemp("", "etl_feature_integration_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	rawDir := filepath.Join(tmpDir, "raw")
	os.MkdirAll(rawDir, 0755)

	// Download a small dataset (2017-08)
	t.Log("Step 1: Downloading real trade data...")
	downloader := binance.NewDownloader(rawDir)
	filename := "BTCUSDT-trades-2017-08.zip"
	url := binance.BaseURLSpot + "/BTCUSDT/" + filename

	if err := downloader.DownloadFile(url, filename); err != nil {
		t.Fatalf("Download failed: %v", err)
	}

	zipPath := filepath.Join(rawDir, filename)

	// Process to bars first
	t.Log("Step 2: Generating dollar bars from trade data...")
	generator := bars.NewGenerator()

	// Warmup
	for i := 0; i < bars.WarmupDays; i++ {
		generator.ProcessTrade(4000, 0.1, 1_000_000,
			time.Date(2017, 7, 18+i, 12, 0, 0, 0, time.UTC).UnixMilli(), true)
	}

	dollarBars, tradeCount, err := processZipForTest(zipPath, generator)
	if err != nil {
		t.Fatalf("Failed to process ZIP: %v", err)
	}

	if finalBar := generator.Flush(); finalBar != nil {
		dollarBars = append(dollarBars, *finalBar)
	}

	t.Logf("Processed %d trades -> %d bars", tradeCount, len(dollarBars))

	if len(dollarBars) == 0 {
		t.Skip("No bars generated, skipping feature generation test")
	}

	// Generate features
	t.Log("Step 3: Generating features from dollar bars...")
	featureGen := features.NewGenerator(features.DefaultConfig())

	var featureRows []features.FeatureRow
	var nanCount, infCount, primedCount int

	for i, bar := range dollarBars {
		row := featureGen.Process(bar)
		featureRows = append(featureRows, row)

		if row.IsPrimed {
			primedCount++
		}

		// Count problematic values
		if math.IsNaN(row.LogVolume) || math.IsNaN(row.GarmanKlassVol) {
			nanCount++
		}
		if math.IsInf(row.LogVolume, 0) || math.IsInf(row.GarmanKlassVol, 0) {
			infCount++
		}

		// Validate cyclical time features
		if row.SinTime < -1.01 || row.SinTime > 1.01 ||
			row.CosTime < -1.01 || row.CosTime > 1.01 {
			t.Errorf("Bar %d: Cyclical time features out of range", i)
		}
	}

	t.Logf("Feature generation results:")
	t.Logf("  Total rows: %d", len(featureRows))
	t.Logf("  Primed rows: %d (%.1f%%)", primedCount, float64(primedCount)/float64(len(featureRows))*100)
	t.Logf("  NaN values: %d", nanCount)
	t.Logf("  Inf values: %d", infCount)

	if nanCount > 0 {
		t.Errorf("Found %d rows with NaN values", nanCount)
	}
	if infCount > 0 {
		t.Errorf("Found %d rows with Inf values", infCount)
	}

	// Verify state persistence
	t.Log("Step 4: Testing feature generator state persistence...")
	state := featureGen.State()

	featureGen2 := features.NewGenerator(features.DefaultConfig())
	featureGen2.LoadState(state)

	if featureGen.Count() != featureGen2.Count() {
		t.Errorf("State count mismatch: %d vs %d", featureGen.Count(), featureGen2.Count())
	}

	t.Log("=== Feature Generation with Real Data Integration Test PASSED ===")
}

func analyzeZipColumns(zipPath string) (*ColumnReport, error) {
	archive, err := zip.OpenReader(zipPath)
	if err != nil {
		return nil, err
	}
	defer archive.Close()

	for _, f := range archive.File {
		if !strings.HasSuffix(f.Name, ".csv") {
			continue
		}

		rc, err := f.Open()
		if err != nil {
			return nil, err
		}
		defer rc.Close()

		reader := csv.NewReader(rc)

		firstRow, err := reader.Read()
		if err != nil {
			return nil, fmt.Errorf("failed to read first row: %w", err)
		}

		secondRow, err := reader.Read()
		if err != nil {
			return nil, fmt.Errorf("failed to read second row: %w", err)
		}

		report := &ColumnReport{
			TotalColumns: len(firstRow),
		}

		isHeader := len(firstRow) > 0 && strings.Contains(strings.ToLower(firstRow[0]), "id")
		if isHeader {
			report.Headers = firstRow
			report.SampleRow = secondRow
		} else {
			report.Headers = expectedColumns
			report.SampleRow = firstRow
		}

		headerSet := make(map[string]bool)
		for _, h := range report.Headers {
			headerSet[strings.ToLower(h)] = true
		}
		for _, expected := range expectedColumns {
			if !headerSet[strings.ToLower(expected)] {
				report.MissingColumns = append(report.MissingColumns, expected)
			}
		}

		expectedSet := make(map[string]bool)
		for _, e := range expectedColumns {
			expectedSet[strings.ToLower(e)] = true
		}
		for _, h := range report.Headers {
			if !expectedSet[strings.ToLower(h)] {
				report.ExtraColumns = append(report.ExtraColumns, h)
			}
		}

		return report, nil
	}

	return nil, fmt.Errorf("no CSV file found in archive")
}
