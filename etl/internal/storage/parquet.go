package storage

import (
	"dl-rl-btc-etl/internal/bars"
	"dl-rl-btc-etl/internal/features"
	"dl-rl-btc-etl/internal/tib"

	"github.com/segmentio/parquet-go"
)

// WriteParquet writes a slice of TIB bars to a parquet file.
// Deprecated: Use WriteDollarBars for new code.
func WriteParquet(filename string, tibBars []tib.Bar) error {
	return parquet.WriteFile(filename, tibBars)
}

// WriteDollarBars writes a slice of DynamicDollarBars to a parquet file.
func WriteDollarBars(filename string, dollarBars []bars.DynamicDollarBar) error {
	return parquet.WriteFile(filename, dollarBars)
}

// ReadDollarBars reads DynamicDollarBars from a parquet file.
func ReadDollarBars(filename string) ([]bars.DynamicDollarBar, error) {
	var result []bars.DynamicDollarBar
	rows, err := parquet.ReadFile[bars.DynamicDollarBar](filename)
	if err != nil {
		return nil, err
	}
	result = append(result, rows...)
	return result, nil
}

// WriteFeatures writes a slice of FeatureRows to a parquet file.
func WriteFeatures(filename string, featureRows []features.FeatureRow) error {
	return parquet.WriteFile(filename, featureRows)
}

// ReadFeatures reads FeatureRows from a parquet file.
func ReadFeatures(filename string) ([]features.FeatureRow, error) {
	var result []features.FeatureRow
	rows, err := parquet.ReadFile[features.FeatureRow](filename)
	if err != nil {
		return nil, err
	}
	result = append(result, rows...)
	return result, nil
}
