package config

import (
	"fmt"
	"os"
	"path/filepath"

	"gopkg.in/yaml.v3"
)

// SymbolConfig represents configuration for a single trading symbol.
type SymbolConfig struct {
	Symbol   string `yaml:"symbol"`
	Market   string `yaml:"market"`   // "spot" or "futures"
	Enabled  bool   `yaml:"enabled"`  // Whether to process this symbol
	Priority int    `yaml:"priority"` // 1=primary, 2=secondary, 3=tertiary
}

// ETLConfig holds the complete ETL pipeline configuration.
type ETLConfig struct {
	// Symbol to process (single symbol mode)
	Symbol string
	// Market type
	Market string // "spot" or "futures"
	// Data directories
	RawDir string
	OutDir string
}

// MultiSymbolConfig holds configuration for multi-symbol processing.
type MultiSymbolConfig struct {
	// Symbols to process
	Symbols []SymbolConfig `yaml:"symbols"`

	// Data directories (base paths)
	DataDir     string `yaml:"data_dir"`      // Base data directory
	RawSubDir   string `yaml:"raw_subdir"`    // Subdirectory for raw data
	BarsSubDir  string `yaml:"bars_subdir"`   // Subdirectory for bars
	StateSubDir string `yaml:"state_subdir"`  // Subdirectory for state files

	// External raw data directory (for persistent raw data storage)
	ExternalRawDir string `yaml:"external_raw_dir"` // e.g., /home/kimhoyeon/rawData

	// Bar generation options
	BarsPerDay int `yaml:"bars_per_day"` // Target bars per day (default: 50)

	// Processing options
	Parallel    bool `yaml:"parallel"`     // Process symbols in parallel
	MaxParallel int  `yaml:"max_parallel"` // Max concurrent symbol processing
}

// DefaultMultiSymbolConfig returns default multi-symbol configuration.
func DefaultMultiSymbolConfig() MultiSymbolConfig {
	return MultiSymbolConfig{
		Symbols: []SymbolConfig{
			// Primary assets (high liquidity)
			{Symbol: "BTCUSDT", Market: "futures", Enabled: true, Priority: 1},
			{Symbol: "ETHUSDT", Market: "futures", Enabled: true, Priority: 1},
			// Secondary assets (medium liquidity)
			{Symbol: "XRPUSDT", Market: "futures", Enabled: true, Priority: 2},
			{Symbol: "SOLUSDT", Market: "futures", Enabled: true, Priority: 2},
			{Symbol: "BNBUSDT", Market: "futures", Enabled: true, Priority: 2},
			// Tertiary assets (higher volatility)
			{Symbol: "DOGEUSDT", Market: "futures", Enabled: true, Priority: 3},
			{Symbol: "LTCUSDT", Market: "futures", Enabled: true, Priority: 3},
		},
		DataDir:        "data",
		RawSubDir:      "raw",
		BarsSubDir:     "bars",
		StateSubDir:    "state",
		ExternalRawDir: "/home/kimhoyeon/rawData",
		BarsPerDay:     50,
		Parallel:       false,
		MaxParallel:    3,
	}
}

// GetEnabledSymbols returns only enabled symbols, sorted by priority.
func (c *MultiSymbolConfig) GetEnabledSymbols() []SymbolConfig {
	var enabled []SymbolConfig
	for _, s := range c.Symbols {
		if s.Enabled {
			enabled = append(enabled, s)
		}
	}
	// Sort by priority (1 first, then 2, then 3)
	for i := 0; i < len(enabled)-1; i++ {
		for j := i + 1; j < len(enabled); j++ {
			if enabled[j].Priority < enabled[i].Priority {
				enabled[i], enabled[j] = enabled[j], enabled[i]
			}
		}
	}
	return enabled
}

// GetETLConfig returns ETLConfig for a specific symbol.
func (c *MultiSymbolConfig) GetETLConfig(symbol SymbolConfig) ETLConfig {
	return ETLConfig{
		Symbol: symbol.Symbol,
		Market: symbol.Market,
		RawDir: c.GetRawDir(symbol),
		OutDir: c.GetBarsDir(symbol),
	}
}

// GetRawDir returns the raw data directory for a symbol.
// Structure: data/raw/{market}/{symbol}/
func (c *MultiSymbolConfig) GetRawDir(symbol SymbolConfig) string {
	return filepath.Join(c.DataDir, c.RawSubDir, symbol.Market, symbol.Symbol)
}

// GetExternalRawDir returns the external raw data directory for persistent storage.
// Structure: {ExternalRawDir}/binance/{market}/{symbol}/
func (c *MultiSymbolConfig) GetExternalRawDir(symbol SymbolConfig) string {
	return filepath.Join(c.ExternalRawDir, "binance", symbol.Market, symbol.Symbol)
}

// GetBarsDir returns the bars output directory for a symbol.
// Structure: data/bars-{barsPerDay}/{market}/{symbol}/
func (c *MultiSymbolConfig) GetBarsDir(symbol SymbolConfig) string {
	barsDir := fmt.Sprintf("%s-%d", c.BarsSubDir, c.BarsPerDay)
	return filepath.Join(c.DataDir, barsDir, symbol.Market, symbol.Symbol)
}

// GetStateDir returns the state directory for a symbol.
// Structure: data/state-{barsPerDay}/{market}/{symbol}/
func (c *MultiSymbolConfig) GetStateDir(symbol SymbolConfig) string {
	stateDir := fmt.Sprintf("%s-%d", c.StateSubDir, c.BarsPerDay)
	return filepath.Join(c.DataDir, stateDir, symbol.Market, symbol.Symbol)
}

// GetFeaturesDir returns the features output directory for a symbol.
// Structure: data/features-{barsPerDay}/{market}/{symbol}/
func (c *MultiSymbolConfig) GetFeaturesDir(symbol SymbolConfig) string {
	featuresDir := fmt.Sprintf("features-%d", c.BarsPerDay)
	return filepath.Join(c.DataDir, featuresDir, symbol.Market, symbol.Symbol)
}

// GetBarsStateFile returns the path to the bars generator state file for a symbol.
// Now simplified since we have symbol-specific directories.
func (c *MultiSymbolConfig) GetBarsStateFile(symbol SymbolConfig) string {
	return filepath.Join(c.GetStateDir(symbol), "generator_state.json")
}

// GetFeatureStateFile returns the path to the feature generator state file for a symbol.
// Now simplified since we have symbol-specific directories.
func (c *MultiSymbolConfig) GetFeatureStateFile(symbol SymbolConfig) string {
	return filepath.Join(c.GetStateDir(symbol), "feature_state.json")
}

// LoadFromYAML loads multi-symbol configuration from a YAML file.
func LoadFromYAML(path string) (*MultiSymbolConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	var config MultiSymbolConfig
	if err := yaml.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("failed to parse config file: %w", err)
	}

	// Apply defaults for missing fields
	if config.DataDir == "" {
		config.DataDir = "data"
	}
	if config.RawSubDir == "" {
		config.RawSubDir = "raw"
	}
	if config.BarsSubDir == "" {
		config.BarsSubDir = "bars"
	}
	if config.StateSubDir == "" {
		config.StateSubDir = "state"
	}
	if config.ExternalRawDir == "" {
		config.ExternalRawDir = "/home/kimhoyeon/rawData"
	}
	if config.BarsPerDay <= 0 {
		config.BarsPerDay = 50
	}

	return &config, nil
}

// SaveToYAML saves multi-symbol configuration to a YAML file.
func (c *MultiSymbolConfig) SaveToYAML(path string) error {
	data, err := yaml.Marshal(c)
	if err != nil {
		return fmt.Errorf("failed to marshal config: %w", err)
	}

	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("failed to write config file: %w", err)
	}

	return nil
}

// EnsureDirectories creates all necessary directories for a symbol.
func (c *MultiSymbolConfig) EnsureDirectories(symbol SymbolConfig) error {
	dirs := []string{
		c.GetRawDir(symbol),
		c.GetBarsDir(symbol),
		c.GetStateDir(symbol),
		c.GetFeaturesDir(symbol),
	}

	for _, dir := range dirs {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return fmt.Errorf("failed to create directory %s: %w", dir, err)
		}
	}

	return nil
}

// EnsureExternalRawDir creates the external raw data directory for a symbol.
func (c *MultiSymbolConfig) EnsureExternalRawDir(symbol SymbolConfig) error {
	dir := c.GetExternalRawDir(symbol)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create external raw directory %s: %w", dir, err)
	}
	return nil
}

// DefaultFuturesConfig returns default config for BTCUSDT Futures.
// Kept for backward compatibility.
func DefaultFuturesConfig() ETLConfig {
	return ETLConfig{
		Symbol: "BTCUSDT",
		Market: "futures",
		RawDir: "data/raw/futures",
		OutDir: "data/bars/futures",
	}
}

// DefaultSpotConfig returns default config for BTCUSDT Spot.
// Kept for backward compatibility.
func DefaultSpotConfig() ETLConfig {
	return ETLConfig{
		Symbol: "BTCUSDT",
		Market: "spot",
		RawDir: "data/raw/spot",
		OutDir: "data/bars/spot",
	}
}
