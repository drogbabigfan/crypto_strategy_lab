package main

import (
	"archive/zip"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"dl-rl-btc-etl/internal/backtest"
	"dl-rl-btc-etl/internal/bars"
	"dl-rl-btc-etl/internal/binance"
	"dl-rl-btc-etl/internal/config"
	"dl-rl-btc-etl/internal/features"
	"dl-rl-btc-etl/internal/storage"
)

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	switch os.Args[1] {
	case "download":
		runDownload()
	case "bars":
		runBarsETL()
	case "features":
		runFeatureGeneration()
	case "backtest":
		runBacktest()
	case "config":
		generateDefaultConfig()
	case "help", "-h", "--help":
		printUsage()
	default:
		log.Printf("Unknown command: %s", os.Args[1])
		printUsage()
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Println("Usage: etl <command> [options]")
	fmt.Println("")
	fmt.Println("Commands:")
	fmt.Println("  download  Download raw trade data from Binance (persistent storage)")
	fmt.Println("  bars      Generate Dollar Bars from downloaded raw data")
	fmt.Println("  features  Generate features from existing Dollar Bars")
	fmt.Println("  backtest  Run backtest with signals and features")
	fmt.Println("  config    Generate default config.yaml")
	fmt.Println("  help      Show this help message")
	fmt.Println("")
	fmt.Println("Options for download:")
	fmt.Println("  -config <path>    Path to config.yaml (default: config.yaml)")
	fmt.Println("  -symbol <symbol>  Process only this symbol (overrides config)")
	fmt.Println("  -parallel         Download symbols in parallel")
	fmt.Println("")
	fmt.Println("Options for bars:")
	fmt.Println("  -config <path>       Path to config.yaml (default: config.yaml)")
	fmt.Println("  -symbol <symbol>     Process only this symbol (overrides config)")
	fmt.Println("  -bars-per-day <n>    Target bars per day (default: 50)")
	fmt.Println("  -parallel            Process symbols in parallel")
	fmt.Println("")
	fmt.Println("Options for features:")
	fmt.Println("  -config <path>    Path to config.yaml (default: config.yaml)")
	fmt.Println("  -symbol <symbol>  Process only this symbol (overrides config)")
	fmt.Println("  -parallel         Process symbols in parallel")
	fmt.Println("")
	fmt.Println("Options for backtest:")
	fmt.Println("  -signals <path>   Path to signals parquet file (required)")
	fmt.Println("  -features <path>  Path to features parquet file (required)")
	fmt.Println("  -config <path>    Path to backtest config JSON file")
	fmt.Println("  -output <path>    Output JSON file path")
	fmt.Println("  -sl <mult>        Stop loss multiplier")
	fmt.Println("  -pt <mult>        Profit target multiplier")
	fmt.Println("  -max-hold <bars>  Maximum holding bars")
	fmt.Println("  -equity           Include equity curve in output")
	fmt.Println("  -quiet            Suppress summary output")
}

func generateDefaultConfig() {
	cfg := config.DefaultMultiSymbolConfig()
	if err := cfg.SaveToYAML("config.yaml"); err != nil {
		log.Fatalf("Failed to save config: %v", err)
	}
	log.Println("Generated default config.yaml")
}

// parseFlags parses command line flags and returns configuration
func parseFlags() (*config.MultiSymbolConfig, string, bool, int) {
	// Define flags
	configPath := flag.String("config", "config.yaml", "Path to config.yaml")
	symbolOverride := flag.String("symbol", "", "Process only this symbol")
	parallel := flag.Bool("parallel", false, "Process symbols in parallel")
	barsPerDay := flag.Int("bars-per-day", 0, "Target bars per day (0 = use config default)")

	// Parse flags (skip command name)
	flag.CommandLine.Parse(os.Args[2:])

	// Try to load config from file
	var cfg *config.MultiSymbolConfig
	if _, err := os.Stat(*configPath); err == nil {
		loadedCfg, err := config.LoadFromYAML(*configPath)
		if err != nil {
			log.Printf("Warning: failed to load %s: %v, using defaults", *configPath, err)
			defaultCfg := config.DefaultMultiSymbolConfig()
			cfg = &defaultCfg
		} else {
			cfg = loadedCfg
			log.Printf("Loaded config from %s", *configPath)
		}
	} else {
		// Use default config
		defaultCfg := config.DefaultMultiSymbolConfig()
		cfg = &defaultCfg
		log.Println("Using default multi-symbol configuration")
	}

	// Override parallel setting if specified
	if *parallel {
		cfg.Parallel = true
	}

	// Override bars per day if specified
	if *barsPerDay > 0 {
		cfg.BarsPerDay = *barsPerDay
	}

	return cfg, *symbolOverride, cfg.Parallel, cfg.BarsPerDay
}

func runDownload() {
	cfg, symbolOverride, parallel, _ := parseFlags()

	// Get symbols to process
	var symbols []config.SymbolConfig
	if symbolOverride != "" {
		for _, s := range cfg.Symbols {
			if s.Symbol == symbolOverride {
				symbols = append(symbols, s)
				break
			}
		}
		if len(symbols) == 0 {
			log.Fatalf("Symbol %s not found in configuration", symbolOverride)
		}
	} else {
		symbols = cfg.GetEnabledSymbols()
	}

	log.Printf("=== Download Pipeline: Downloading %d symbols ===", len(symbols))
	log.Printf("Raw data directory: %s", cfg.ExternalRawDir)

	if parallel && len(symbols) > 1 {
		runDownloadParallel(cfg, symbols)
	} else {
		runDownloadSequential(cfg, symbols)
	}

	log.Println("Download Pipeline Finished.")
}

func runDownloadSequential(cfg *config.MultiSymbolConfig, symbols []config.SymbolConfig) {
	for _, symbol := range symbols {
		log.Printf("\n=== Downloading %s (%s) ===", symbol.Symbol, symbol.Market)
		if err := downloadSymbol(cfg, symbol); err != nil {
			log.Printf("Error downloading %s: %v", symbol.Symbol, err)
		}
	}
}

func runDownloadParallel(cfg *config.MultiSymbolConfig, symbols []config.SymbolConfig) {
	maxWorkers := cfg.MaxParallel
	if maxWorkers <= 0 {
		maxWorkers = 3
	}

	sem := make(chan struct{}, maxWorkers)
	var wg sync.WaitGroup

	for _, symbol := range symbols {
		wg.Add(1)
		sem <- struct{}{}

		go func(s config.SymbolConfig) {
			defer wg.Done()
			defer func() { <-sem }()

			log.Printf("[%s] Starting download...", s.Symbol)
			if err := downloadSymbol(cfg, s); err != nil {
				log.Printf("[%s] Download error: %v", s.Symbol, err)
			} else {
				log.Printf("[%s] Download completed", s.Symbol)
			}
		}(symbol)
	}

	wg.Wait()
}

func downloadSymbol(cfg *config.MultiSymbolConfig, symbol config.SymbolConfig) error {
	// Ensure external raw directory exists
	if err := cfg.EnsureExternalRawDir(symbol); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}

	rawDir := cfg.GetExternalRawDir(symbol)
	downloader := binance.NewDownloader(rawDir)

	// Determine date range
	var startDate time.Time
	if symbol.Market == "futures" {
		startDate = binance.FuturesStartDate
	} else {
		startDate = binance.SpotStartDate
	}
	endDate := time.Now()

	// Generate month range
	current := time.Date(startDate.Year(), startDate.Month(), 1, 0, 0, 0, 0, time.UTC)
	end := time.Date(endDate.Year(), endDate.Month(), 1, 0, 0, 0, 0, time.UTC)

	downloadCount := 0
	skipCount := 0

	for !current.After(end) {
		year := current.Year()
		month := int(current.Month())

		// Skip future months
		now := time.Now()
		if current.Year() > now.Year() || (current.Year() == now.Year() && current.Month() > now.Month()) {
			break
		}

		// Generate URL and local path
		var baseURL string
		if symbol.Market == "futures" {
			baseURL = binance.BaseURLFutures
		} else {
			baseURL = binance.BaseURLSpot
		}

		filename := fmt.Sprintf("%s-trades-%d-%02d.zip", symbol.Symbol, year, month)
		url := fmt.Sprintf("%s/%s/%s", baseURL, symbol.Symbol, filename)
		localPath := filepath.Join(rawDir, filename)

		// Download if not exists
		if _, err := os.Stat(localPath); os.IsNotExist(err) {
			log.Printf("  [%s] Downloading %s...", symbol.Symbol, filename)
			if err := downloader.DownloadFile(url, filename); err != nil {
				log.Printf("  [%s] Failed to download %s: %v", symbol.Symbol, filename, err)
			} else {
				downloadCount++
			}
		} else {
			skipCount++
		}

		current = current.AddDate(0, 1, 0)
	}

	log.Printf("[%s] Downloaded %d files, skipped %d (already exist)", symbol.Symbol, downloadCount, skipCount)
	return nil
}

func runBarsETL() {
	cfg, symbolOverride, parallel, barsPerDay := parseFlags()

	// Get symbols to process
	var symbols []config.SymbolConfig
	if symbolOverride != "" {
		// Process only specified symbol
		for _, s := range cfg.Symbols {
			if s.Symbol == symbolOverride {
				symbols = append(symbols, s)
				break
			}
		}
		if len(symbols) == 0 {
			log.Fatalf("Symbol %s not found in configuration", symbolOverride)
		}
	} else {
		symbols = cfg.GetEnabledSymbols()
	}

	log.Printf("=== Bar Generation Pipeline: Processing %d symbols ===", len(symbols))
	log.Printf("Raw data directory: %s", cfg.ExternalRawDir)
	log.Printf("Target bars per day: %d", barsPerDay)
	for _, s := range symbols {
		log.Printf("  - %s (%s, priority %d)", s.Symbol, s.Market, s.Priority)
	}

	if parallel && len(symbols) > 1 {
		runBarsParallel(cfg, symbols)
	} else {
		runBarsSequential(cfg, symbols)
	}

	log.Println("Bar Generation Pipeline Finished.")
}

func runBarsSequential(cfg *config.MultiSymbolConfig, symbols []config.SymbolConfig) {
	for _, symbol := range symbols {
		log.Printf("\n=== Processing %s (%s) ===", symbol.Symbol, symbol.Market)
		if err := processSymbol(cfg, symbol); err != nil {
			log.Printf("Error processing %s: %v", symbol.Symbol, err)
		}
	}
}

func runBarsParallel(cfg *config.MultiSymbolConfig, symbols []config.SymbolConfig) {
	maxWorkers := cfg.MaxParallel
	if maxWorkers <= 0 {
		maxWorkers = 3
	}

	sem := make(chan struct{}, maxWorkers)
	var wg sync.WaitGroup

	for _, symbol := range symbols {
		wg.Add(1)
		sem <- struct{}{} // Acquire semaphore

		go func(s config.SymbolConfig) {
			defer wg.Done()
			defer func() { <-sem }() // Release semaphore

			log.Printf("[%s] Starting processing...", s.Symbol)
			if err := processSymbol(cfg, s); err != nil {
				log.Printf("[%s] Error: %v", s.Symbol, err)
			} else {
				log.Printf("[%s] Completed successfully", s.Symbol)
			}
		}(symbol)
	}

	wg.Wait()
}

func processSymbol(cfg *config.MultiSymbolConfig, symbol config.SymbolConfig) error {
	// Ensure directories exist
	if err := cfg.EnsureDirectories(symbol); err != nil {
		return fmt.Errorf("failed to create directories: %w", err)
	}

	// Get paths
	etlCfg := cfg.GetETLConfig(symbol)
	stateDir := cfg.GetStateDir(symbol)
	rawDir := cfg.GetExternalRawDir(symbol)

	// Initialize components with configurable bars per day
	generator := bars.NewGeneratorWithBarsPerDay(cfg.BarsPerDay)

	// Try to migrate legacy state if exists
	if migrated, err := bars.MigrateState(etlCfg.OutDir, stateDir, symbol.Symbol); err != nil {
		log.Printf("[%s] Warning: state migration failed: %v", symbol.Symbol, err)
	} else if migrated {
		log.Printf("[%s] Migrated legacy state to new location", symbol.Symbol)
	}

	// Load saved state if exists (for continuity across runs)
	savedState, err := bars.LoadStateForSymbol(stateDir, symbol.Symbol)
	if err != nil {
		// State validation failed - corrupted state detected
		log.Printf("[%s] ⚠️ CORRUPTED STATE DETECTED: %v", symbol.Symbol, err)
		log.Printf("[%s] Deleting corrupted state and starting fresh...", symbol.Symbol)
		if delErr := bars.DeleteStateForSymbol(stateDir, symbol.Symbol); delErr != nil {
			log.Printf("[%s] Warning: failed to delete corrupted state: %v", symbol.Symbol, delErr)
		}
		// Continue with fresh generator (savedState remains nil)
	} else if savedState != nil {
		generator.ImportState(*savedState)
		log.Printf("[%s] Restored generator state from checkpoint", symbol.Symbol)

		// Check for partial month and handle it
		if partial := generator.GetCurrentMonth(); partial != "" {
			log.Printf("[%s] Detected partial month: %s - will reprocess", symbol.Symbol, partial)
			// Delete partial month's file if exists
			partialFile := filepath.Join(etlCfg.OutDir, fmt.Sprintf("%s-bars-%s.parquet", symbol.Symbol, partial))
			os.Remove(partialFile)
			// Restore from previous month's checkpoint
			if prevCp, exists := generator.GetPreviousMonthCheckpoint(partial); exists {
				generator.RestoreFromCheckpoint(prevCp)
				log.Printf("[%s] Restored state from %s checkpoint", symbol.Symbol, partial)
			}
			generator.DeleteMonthCheckpoint(partial)
		}

		if generator.IsWarmup() {
			log.Printf("[%s] Still in warmup: %d days remaining", symbol.Symbol, generator.GetWarmupDaysRemaining())
		} else {
			log.Printf("[%s] Threshold: $%.2f", symbol.Symbol, generator.GetCurrentThreshold())
		}
	}

	// Determine date range
	var startDate time.Time
	if symbol.Market == "futures" {
		startDate = binance.FuturesStartDate
	} else {
		startDate = binance.SpotStartDate
	}
	endDate := time.Now()

	// Generate month range
	current := time.Date(startDate.Year(), startDate.Month(), 1, 0, 0, 0, 0, time.UTC)
	end := time.Date(endDate.Year(), endDate.Month(), 1, 0, 0, 0, 0, time.UTC)

	// Track bars and metadata for current batch
	barsByMonth := make(map[string][]bars.DynamicDollarBar)
	metaByMonth := make(map[string]*monthMeta)
	var processedMonths []string

	for !current.After(end) {
		year := current.Year()
		month := int(current.Month())
		monthKey := fmt.Sprintf("%d-%02d", year, month)

		// Skip future months
		now := time.Now()
		if current.Year() > now.Year() || (current.Year() == now.Year() && current.Month() > now.Month()) {
			break
		}

		// Check if month is already complete
		needsReprocessing := false
		if generator.IsMonthComplete(monthKey) {
			// Verify file exists and matches checkpoint
			outFile := filepath.Join(etlCfg.OutDir, fmt.Sprintf("%s-bars-%s.parquet", symbol.Symbol, monthKey))
			if fileInfo, err := os.Stat(outFile); err == nil {
				cp, _ := generator.GetMonthCheckpoint(monthKey)
				log.Printf("[%s] %s: Already complete (%d bars), skipping", symbol.Symbol, monthKey, cp.BarCount)
				_ = fileInfo // File exists, checkpoint valid
				current = current.AddDate(0, 1, 0)
				continue
			}
			// File missing but marked complete - need to reprocess
			log.Printf("[%s] %s: Marked complete but file missing, will reprocess", symbol.Symbol, monthKey)
			generator.DeleteMonthCheckpoint(monthKey)
			needsReprocessing = true
		}

		// === STATELESS START PRINCIPLE ===
		// Only restore from checkpoint when REPROCESSING (file was deleted).
		// For continuous processing, the current state is already correct.
		if needsReprocessing {
			if restored, prevMonth := generator.PrepareForMonth(monthKey); restored {
				log.Printf("[%s] %s: Restored clean state from %s checkpoint for reprocessing", symbol.Symbol, monthKey, prevMonth)
			}
		}

		// Validate state before processing (Fail-Fast)
		if err := generator.ValidateStateForMonth(monthKey, 0); err != nil {
			log.Printf("[%s] %s: State validation failed: %v", symbol.Symbol, monthKey, err)
			log.Printf("[%s] %s: Resetting state and reprocessing from scratch", symbol.Symbol, monthKey)
			// Reset to fresh state for this month
			generator.DeleteMonthCheckpoint(monthKey)
			if restored, prevMonth := generator.PrepareForMonth(monthKey); restored {
				log.Printf("[%s] %s: Restored from %s after validation failure", symbol.Symbol, monthKey, prevMonth)
			}
		}

		log.Printf("[%s] Processing %d-%02d...", symbol.Symbol, year, month)

		// Mark as currently processing (partial state)
		generator.SetCurrentMonth(monthKey)
		generator.MarkMonthPartial(monthKey)

		// Process month (returns bars generated in this month)
		monthBars, tradeCount, firstTradeID, lastTradeID, err := processMonthWithMeta(etlCfg, rawDir, generator, year, month)
		if err != nil {
			log.Printf("[%s] %d-%02d failed: %v", symbol.Symbol, year, month, err)
			current = current.AddDate(0, 1, 0)
			continue
		}

		if generator.IsWarmup() {
			log.Printf("[%s] %d-%02d: Warmup mode (%d days remaining), processed %d trades",
				symbol.Symbol, year, month, generator.GetWarmupDaysRemaining(), tradeCount)
		} else {
			log.Printf("[%s] %d-%02d: Processed %d trades -> %d bars (threshold: $%.2f)",
				symbol.Symbol, year, month, tradeCount, len(monthBars), generator.GetCurrentThreshold())
		}

		// Distribute bars to their respective months (by StartTime)
		for _, bar := range monthBars {
			t := time.UnixMilli(bar.StartTime).UTC()
			key := fmt.Sprintf("%d-%02d", t.Year(), t.Month())
			barsByMonth[key] = append(barsByMonth[key], bar)

			// Track metadata
			if metaByMonth[key] == nil {
				metaByMonth[key] = &monthMeta{
					firstTradeID: firstTradeID,
					firstBarTime: bar.StartTime,
				}
			}
			metaByMonth[key].lastTradeID = lastTradeID
			metaByMonth[key].lastBarTime = bar.EndTime
		}

		// Track current month
		processedMonths = append(processedMonths, monthKey)

		// Save and finalize months that are 2+ months behind
		if len(processedMonths) > 2 {
			oldKey := processedMonths[0]
			if oldBars, exists := barsByMonth[oldKey]; exists && len(oldBars) > 0 {
				if err := saveMonthBars(etlCfg, oldKey, oldBars); err != nil {
					log.Printf("[%s] Warning: failed to save %s: %v", symbol.Symbol, oldKey, err)
				} else {
					// Mark as complete with metadata
					meta := metaByMonth[oldKey]
					if meta != nil {
						generator.MarkMonthComplete(oldKey, len(oldBars),
							meta.firstTradeID, meta.lastTradeID,
							meta.firstBarTime, meta.lastBarTime)
					}
				}
			}
			delete(barsByMonth, oldKey)
			delete(metaByMonth, oldKey)
			processedMonths = processedMonths[1:]
		}

		// Clear current processing month (will be set complete when saved)
		generator.SetCurrentMonth("")

		// Save generator state after each month (checkpoint)
		if err := bars.SaveStateForSymbol(stateDir, symbol.Symbol, generator.ExportState()); err != nil {
			log.Printf("[%s] Warning: failed to save state: %v", symbol.Symbol, err)
		}

		current = current.AddDate(0, 1, 0)
	}

	// Flush current day's volume
	generator.FlushDay()

	// Flush any remaining bar
	if finalBar := generator.Flush(); finalBar != nil {
		t := time.UnixMilli(finalBar.StartTime).UTC()
		key := fmt.Sprintf("%d-%02d", t.Year(), t.Month())
		barsByMonth[key] = append(barsByMonth[key], *finalBar)
		if metaByMonth[key] != nil {
			metaByMonth[key].lastBarTime = finalBar.EndTime
		}
		log.Printf("[%s] Flushed final incomplete bar", symbol.Symbol)
	}

	// Save remaining months in memory
	for key, monthBars := range barsByMonth {
		if len(monthBars) > 0 {
			if err := saveMonthBars(etlCfg, key, monthBars); err != nil {
				return fmt.Errorf("failed to save %s: %w", key, err)
			}
			// Mark as complete
			meta := metaByMonth[key]
			if meta != nil {
				generator.MarkMonthComplete(key, len(monthBars),
					meta.firstTradeID, meta.lastTradeID,
					meta.firstBarTime, meta.lastBarTime)
			}
		}
	}

	// Final state save
	if err := bars.SaveStateForSymbol(stateDir, symbol.Symbol, generator.ExportState()); err != nil {
		log.Printf("[%s] Warning: failed to save final state: %v", symbol.Symbol, err)
	}

	return nil
}

type monthMeta struct {
	firstTradeID int64
	lastTradeID  int64
	firstBarTime int64
	lastBarTime  int64
}

// processMonthWithMeta processes a month's trade data from raw files, returning bars and trade ID metadata.
// Raw data is read from external rawDir and preserved for reuse.
func processMonthWithMeta(cfg config.ETLConfig, rawDir string, generator *bars.Generator, year, month int) ([]bars.DynamicDollarBar, int, int64, int64, error) {
	// Find raw ZIP file in external raw directory
	filename := fmt.Sprintf("%s-trades-%d-%02d.zip", cfg.Symbol, year, month)
	localPath := filepath.Join(rawDir, filename)

	// Check if raw file exists
	if _, err := os.Stat(localPath); os.IsNotExist(err) {
		return nil, 0, 0, 0, fmt.Errorf("raw file not found: %s (run 'etl download' first)", localPath)
	}

	// Process ZIP file with metadata tracking (raw file is preserved)
	dollarBars, tradeCount, firstTradeID, lastTradeID, err := processZipFileWithMeta(localPath, generator)
	if err != nil {
		return nil, 0, 0, 0, fmt.Errorf("processing failed: %w", err)
	}

	return dollarBars, tradeCount, firstTradeID, lastTradeID, nil
}

// processZipFileWithMeta processes a ZIP file and returns bars along with first/last trade IDs.
func processZipFileWithMeta(zipPath string, generator *bars.Generator) ([]bars.DynamicDollarBar, int, int64, int64, error) {
	archive, err := zip.OpenReader(zipPath)
	if err != nil {
		return nil, 0, 0, 0, fmt.Errorf("failed to open zip: %w", err)
	}
	defer archive.Close()

	var allBars []bars.DynamicDollarBar
	totalCount := 0
	var firstTradeID, lastTradeID int64

	for _, f := range archive.File {
		if !strings.HasSuffix(f.Name, ".csv") {
			continue
		}

		rc, err := f.Open()
		if err != nil {
			return nil, 0, 0, 0, fmt.Errorf("failed to open file in zip: %w", err)
		}

		fileBars, count, fileFirstID, fileLastID, err := processCSVWithMeta(rc, generator)
		rc.Close()

		if err != nil {
			return nil, totalCount, firstTradeID, lastTradeID, err
		}

		allBars = append(allBars, fileBars...)
		totalCount += count

		// Track first/last trade IDs
		if firstTradeID == 0 || (fileFirstID > 0 && fileFirstID < firstTradeID) {
			firstTradeID = fileFirstID
		}
		if fileLastID > lastTradeID {
			lastTradeID = fileLastID
		}
	}

	return allBars, totalCount, firstTradeID, lastTradeID, nil
}

// processCSVWithMeta processes a CSV and returns bars along with first/last trade IDs.
func processCSVWithMeta(r io.Reader, generator *bars.Generator) ([]bars.DynamicDollarBar, int, int64, int64, error) {
	reader := csv.NewReader(r)
	var resultBars []bars.DynamicDollarBar
	first := true
	count := 0
	skipped := 0
	var firstTradeID, lastTradeID int64

	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, count, firstTradeID, lastTradeID, fmt.Errorf("csv read error: %w", err)
		}

		// Skip header if present
		if first {
			first = false
			if len(record) > 0 && strings.Contains(strings.ToLower(record[0]), "id") {
				continue
			}
		}

		// Parse trade
		trade, err := binance.ParseTrade(record)
		if err != nil {
			continue // Skip malformed records
		}

		// Track first trade ID
		if firstTradeID == 0 {
			firstTradeID = trade.ID
		}
		lastTradeID = trade.ID

		// Early skip check for already-processed trades (crash recovery)
		if generator.ShouldSkipTrade(trade.ID) {
			skipped++
			continue
		}

		// Process through Dollar Bar generator with trade ID for crash recovery
		// Using Safe version for Fail-Fast state corruption detection
		bar, processErr := generator.ProcessTradeWithIDSafe(
			trade.ID,
			trade.Price,
			trade.Quantity,
			trade.QuoteQuantity,
			trade.Time,
			trade.IsAggressorBuy,
		)

		if processErr != nil {
			// State corruption detected - fail fast
			return resultBars, count, firstTradeID, lastTradeID, fmt.Errorf("state corruption detected: %w", processErr)
		}

		if bar != nil {
			resultBars = append(resultBars, *bar)
		}
		count++
	}

	if skipped > 0 {
		log.Printf("    Skipped %d already-processed trades (resume mode)", skipped)
	}

	return resultBars, count, firstTradeID, lastTradeID, nil
}

// saveMonthBars saves a single month's bars to parquet.
func saveMonthBars(cfg config.ETLConfig, monthKey string, monthBars []bars.DynamicDollarBar) error {
	outFile := filepath.Join(cfg.OutDir, fmt.Sprintf("%s-bars-%s.parquet", cfg.Symbol, monthKey))
	if err := storage.WriteDollarBars(outFile, monthBars); err != nil {
		return fmt.Errorf("failed to write %s: %w", monthKey, err)
	}
	log.Printf("  [%s] Saved %d bars to %s", cfg.Symbol, len(monthBars), outFile)
	return nil
}

// =============================================================================
// Feature Generation
// =============================================================================

func runFeatureGeneration() {
	cfg, symbolOverride, parallel, _ := parseFlags()

	// Get symbols to process
	var symbols []config.SymbolConfig
	if symbolOverride != "" {
		for _, s := range cfg.Symbols {
			if s.Symbol == symbolOverride {
				symbols = append(symbols, s)
				break
			}
		}
		if len(symbols) == 0 {
			log.Fatalf("Symbol %s not found in configuration", symbolOverride)
		}
	} else {
		symbols = cfg.GetEnabledSymbols()
	}

	log.Printf("=== Feature Generation: Processing %d symbols ===", len(symbols))
	log.Printf("Bars per day: %d", cfg.BarsPerDay)
	log.Printf("Reading from: data/bars-%d/", cfg.BarsPerDay)
	log.Printf("Writing to: data/features-%d/", cfg.BarsPerDay)

	if parallel && len(symbols) > 1 {
		runFeaturesParallel(cfg, symbols)
	} else {
		runFeaturesSequential(cfg, symbols)
	}

	log.Println("Feature Generation Finished.")
}

func runFeaturesSequential(cfg *config.MultiSymbolConfig, symbols []config.SymbolConfig) {
	for _, symbol := range symbols {
		log.Printf("\n=== Generating Features for %s ===", symbol.Symbol)
		if err := generateFeaturesForSymbol(cfg, symbol); err != nil {
			log.Printf("Error generating features for %s: %v", symbol.Symbol, err)
		}
	}
}

func runFeaturesParallel(cfg *config.MultiSymbolConfig, symbols []config.SymbolConfig) {
	maxWorkers := cfg.MaxParallel
	if maxWorkers <= 0 {
		maxWorkers = 3
	}

	sem := make(chan struct{}, maxWorkers)
	var wg sync.WaitGroup

	for _, symbol := range symbols {
		wg.Add(1)
		sem <- struct{}{}

		go func(s config.SymbolConfig) {
			defer wg.Done()
			defer func() { <-sem }()

			log.Printf("[%s] Starting feature generation...", s.Symbol)
			if err := generateFeaturesForSymbol(cfg, s); err != nil {
				log.Printf("[%s] Feature generation error: %v", s.Symbol, err)
			} else {
				log.Printf("[%s] Feature generation completed", s.Symbol)
			}
		}(symbol)
	}

	wg.Wait()
}

func generateFeaturesForSymbol(cfg *config.MultiSymbolConfig, symbol config.SymbolConfig) error {
	// Ensure directories exist
	if err := cfg.EnsureDirectories(symbol); err != nil {
		return fmt.Errorf("failed to create directories: %w", err)
	}

	etlCfg := cfg.GetETLConfig(symbol)
	featuresDir := cfg.GetFeaturesDir(symbol)
	stateDir := cfg.GetStateDir(symbol)

	// Find all bar parquet files
	pattern := filepath.Join(etlCfg.OutDir, fmt.Sprintf("%s-bars-*.parquet", symbol.Symbol))
	barFiles, err := filepath.Glob(pattern)
	if err != nil {
		return fmt.Errorf("failed to glob bar files: %w", err)
	}

	if len(barFiles) == 0 {
		return fmt.Errorf("no bar files found in %s", etlCfg.OutDir)
	}

	// Sort by filename (chronological order)
	sort.Strings(barFiles)
	log.Printf("[%s] Found %d bar files to process", symbol.Symbol, len(barFiles))

	// Initialize feature generator
	featureGen := features.NewGenerator(features.DefaultConfig())

	// Load saved feature state if exists (in symbol-specific directory)
	stateFile := filepath.Join(stateDir, "feature_state.json")
	if state, err := loadFeatureState(stateFile); err == nil {
		featureGen.LoadState(state)
		log.Printf("[%s] Restored feature generator state (count: %d)", symbol.Symbol, featureGen.Count())
	}

	// Process each bar file
	for _, barFile := range barFiles {
		// Extract month key from filename
		base := filepath.Base(barFile)
		// Format: BTCUSDT-bars-2024-01.parquet
		parts := strings.Split(base, "-")
		if len(parts) < 4 {
			log.Printf("[%s] Skipping invalid filename: %s", symbol.Symbol, base)
			continue
		}
		monthKey := parts[2] + "-" + strings.TrimSuffix(parts[3], ".parquet")

		// Check if features already exist
		featureFile := filepath.Join(featuresDir, fmt.Sprintf("%s-features-%s.parquet", symbol.Symbol, monthKey))
		if _, err := os.Stat(featureFile); err == nil {
			log.Printf("[%s] %s: Already exists, skipping", symbol.Symbol, monthKey)
			continue
		}

		// Read bars
		dollarBars, err := storage.ReadDollarBars(barFile)
		if err != nil {
			log.Printf("[%s] %s: Failed to read bars: %v", symbol.Symbol, monthKey, err)
			continue
		}

		if len(dollarBars) == 0 {
			log.Printf("[%s] %s: No bars, skipping", symbol.Symbol, monthKey)
			continue
		}

		// Generate features
		featureRows := make([]features.FeatureRow, 0, len(dollarBars))
		for _, bar := range dollarBars {
			row := featureGen.Process(bar)
			featureRows = append(featureRows, row)
		}

		// Write features
		if err := storage.WriteFeatures(featureFile, featureRows); err != nil {
			return fmt.Errorf("failed to write features for %s: %w", monthKey, err)
		}

		primedCount := 0
		for _, row := range featureRows {
			if row.IsPrimed {
				primedCount++
			}
		}
		log.Printf("[%s] %s: Generated %d features (%d primed)", symbol.Symbol, monthKey, len(featureRows), primedCount)

		// Save state after each month
		if err := saveFeatureState(stateFile, featureGen.State()); err != nil {
			log.Printf("[%s] Warning: failed to save state: %v", symbol.Symbol, err)
		}
	}

	return nil
}

func loadFeatureState(filename string) (features.GeneratorState, error) {
	var state features.GeneratorState
	data, err := os.ReadFile(filename)
	if err != nil {
		return state, err
	}
	if err := json.Unmarshal(data, &state); err != nil {
		return state, err
	}
	return state, nil
}

func saveFeatureState(filename string, state features.GeneratorState) error {
	// Ensure directory exists
	dir := filepath.Dir(filename)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create state directory: %w", err)
	}

	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filename, data, 0644)
}

// =============================================================================
// Backtest
// =============================================================================

func runBacktest() {
	// Define flags
	fs := flag.NewFlagSet("backtest", flag.ExitOnError)
	signalsPath := fs.String("signals", "", "Path to signals parquet file (required)")
	featuresPath := fs.String("features", "", "Path to features parquet file (required)")
	configPath := fs.String("config", "", "Path to config JSON file")
	outputPath := fs.String("output", "", "Path to output JSON file")
	includeEquity := fs.Bool("equity", false, "Include equity curve in output")
	quiet := fs.Bool("quiet", false, "Suppress summary output")
	slMult := fs.Float64("sl", 0, "Stop loss multiplier")
	ptMult := fs.Float64("pt", 0, "Profit target multiplier")
	maxHold := fs.Int("max-hold", 0, "Max holding bars")

	fs.Parse(os.Args[2:])

	// Validate required flags
	if *signalsPath == "" || *featuresPath == "" {
		fmt.Fprintln(os.Stderr, "Error: -signals and -features are required")
		fs.Usage()
		os.Exit(1)
	}

	// Load or create config
	var cfg backtest.Config
	var err error

	if *configPath != "" {
		cfg, err = backtest.ConfigFromJSON(*configPath)
		if err != nil {
			log.Fatalf("Error loading config: %v", err)
		}
	} else {
		cfg = backtest.DefaultConfig()
	}

	// Apply overrides
	if *slMult > 0 {
		cfg.SLMult = *slMult
	}
	if *ptMult > 0 {
		cfg.PTMult = *ptMult
	}
	if *maxHold > 0 {
		cfg.MaxHoldBars = *maxHold
	}

	// Create runner
	runner, err := backtest.NewRunner(cfg)
	if err != nil {
		log.Fatalf("Error creating runner: %v", err)
	}

	// Run backtest
	result, err := runner.Run(*signalsPath, *featuresPath)
	if err != nil {
		log.Fatalf("Error running backtest: %v", err)
	}

	// Output results
	if !*quiet {
		result.PrintSummary()
		fmt.Println()
	}

	if *outputPath != "" {
		err = result.WriteJSON(*outputPath, *includeEquity)
		if err != nil {
			log.Fatalf("Error writing output: %v", err)
		}
		if !*quiet {
			log.Printf("Results written to: %s", *outputPath)
		}
	} else {
		jsonResult := result.ToJSON(*includeEquity)
		data, err := json.MarshalIndent(jsonResult, "", "  ")
		if err != nil {
			log.Fatalf("Error marshaling JSON: %v", err)
		}
		fmt.Println(string(data))
	}
}
