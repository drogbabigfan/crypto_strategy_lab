package storage

import (
	"dl-rl-btc-etl/internal/tib"
	"os"
	"path/filepath"
	"testing"

	"github.com/segmentio/parquet-go"
)

func TestWriteReadParquet(t *testing.T) {
	// Create dummy TIBs
	bars := []tib.Bar{
		{Timestamp: 1000, Open: 100.1, High: 100.2, Low: 100.0, Close: 100.15, Volume: 10.5, TickCount: 50},
		{Timestamp: 2000, Open: 100.15, High: 100.3, Low: 100.1, Close: 100.25, Volume: 20.0, TickCount: 60},
	}

	tmpDir := t.TempDir()
	filePath := filepath.Join(tmpDir, "test.parquet")

	// Write
	err := WriteParquet(filePath, bars)
	if err != nil {
		t.Fatalf("WriteParquet failed: %v", err)
	}

	// Read back (using parquet-go generic reader for verification)
	f, err := os.Open(filePath)
	if err != nil {
		t.Fatalf("Failed to open file: %v", err)
	}
	defer f.Close()

	// Re-open for parquet.Read (convenience)
	readBars, err := parquet.ReadFile[tib.Bar](filePath)
	if err != nil {
		t.Fatalf("Failed to read parquet: %v", err)
	}

	if len(readBars) != 2 {
		t.Fatalf("Expected 2 bars, got %d", len(readBars))
	}

	if readBars[0].Close != 100.15 {
		t.Errorf("Expected close 100.15, got %f", readBars[0].Close)
	}
}
