package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"dl-rl-btc-etl/internal/backtest"
)

func main() {
	// Define flags
	signalsPath := flag.String("signals", "", "Path to signals parquet file (required)")
	featuresPath := flag.String("features", "", "Path to features parquet file (required)")
	configPath := flag.String("config", "", "Path to config JSON file (optional, uses defaults if not provided)")
	outputPath := flag.String("output", "", "Path to output JSON file (optional, prints to stdout if not provided)")
	includeEquity := flag.Bool("equity", false, "Include equity curve in output")
	quiet := flag.Bool("quiet", false, "Suppress summary output (only output JSON)")

	// Config overrides (optional)
	exitMode := flag.String("exit-mode", "", "Exit mode: 'tbm' (Triple Barrier) or 'signal' (opposite signal)")
	slMult := flag.Float64("sl", 0, "Stop loss multiplier (overrides config, TBM mode only)")
	ptMult := flag.Float64("pt", 0, "Profit target multiplier (overrides config, TBM mode only)")
	maxHold := flag.Int("max-hold", 0, "Max holding bars (overrides config, TBM mode only)")
	compounding := flag.Bool("compounding", false, "Use compounding returns (reinvest profits)")
	maxLoss := flag.Float64("max-loss", 0.5, "Max loss before stopping backtest (0.5 = 50%)")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Go Backtester - Fast backtest execution engine\n\n")
		fmt.Fprintf(os.Stderr, "Usage: %s [options]\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Required:\n")
		fmt.Fprintf(os.Stderr, "  -signals    Path to signals parquet file\n")
		fmt.Fprintf(os.Stderr, "  -features   Path to features parquet file\n\n")
		fmt.Fprintf(os.Stderr, "Exit Modes:\n")
		fmt.Fprintf(os.Stderr, "  tbm     Triple Barrier Method - exit on TP/SL/Timeout (default)\n")
		fmt.Fprintf(os.Stderr, "  signal  Signal-based - exit on opposite signal (Long->Short, Short->Long)\n\n")
		fmt.Fprintf(os.Stderr, "Optional:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExamples:\n")
		fmt.Fprintf(os.Stderr, "  # TBM mode (default)\n")
		fmt.Fprintf(os.Stderr, "  %s -signals signals.parquet -features features.parquet -sl 2.0 -pt 2.5\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "\n  # Signal mode (no TP/SL needed)\n")
		fmt.Fprintf(os.Stderr, "  %s -signals signals.parquet -features features.parquet -exit-mode signal\n", os.Args[0])
	}

	flag.Parse()

	// Validate required flags
	if *signalsPath == "" || *featuresPath == "" {
		fmt.Fprintln(os.Stderr, "Error: -signals and -features are required")
		flag.Usage()
		os.Exit(1)
	}

	// Load or create config
	var cfg backtest.Config
	var err error

	if *configPath != "" {
		cfg, err = backtest.ConfigFromJSON(*configPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error loading config: %v\n", err)
			os.Exit(1)
		}
	} else {
		cfg = backtest.DefaultConfig()
	}

	// Apply overrides
	if *exitMode != "" {
		cfg.ExitMode = backtest.ExitMode(*exitMode)
	}
	if *slMult > 0 {
		cfg.SLMult = *slMult
	}
	if *ptMult > 0 {
		cfg.PTMult = *ptMult
	}
	if *maxHold > 0 {
		cfg.MaxHoldBars = *maxHold
	}
	if *compounding {
		cfg.Compounding = true
	}
	if *maxLoss > 0 {
		cfg.MaxLossPct = *maxLoss
	}

	// Create runner
	runner, err := backtest.NewRunner(cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating runner: %v\n", err)
		os.Exit(1)
	}

	// Run backtest
	result, err := runner.Run(*signalsPath, *featuresPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error running backtest: %v\n", err)
		os.Exit(1)
	}

	// Output results
	if !*quiet {
		result.PrintSummary()
		fmt.Println()
	}

	if *outputPath != "" {
		// Write to file
		err = result.WriteJSON(*outputPath, *includeEquity)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error writing output: %v\n", err)
			os.Exit(1)
		}
		if !*quiet {
			fmt.Printf("Results written to: %s\n", *outputPath)
		}
	} else {
		// Output JSON to stdout
		jsonResult := result.ToJSON(*includeEquity)
		data, err := json.MarshalIndent(jsonResult, "", "  ")
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error marshaling JSON: %v\n", err)
			os.Exit(1)
		}
		fmt.Println(string(data))
	}
}
