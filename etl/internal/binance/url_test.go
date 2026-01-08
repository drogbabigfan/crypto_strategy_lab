package binance

import (
	"testing"
	"time"
)

func TestGenerateMonthlyURLs(t *testing.T) {
	// Target: 2024-02-01 to 2024-03-01
	// Warmup: 30 days -> Should include 2024-01
	
	symbol := "BTCUSDT"
	startDate := time.Date(2024, 2, 1, 0, 0, 0, 0, time.UTC)
	endDate := time.Date(2024, 3, 1, 0, 0, 0, 0, time.UTC)
	warmupDays := 30
	
	expectedBase := "https://data.binance.vision/data/spot/monthly/trades"
	
	urls, err := GenerateMonthlyURLs(symbol, startDate, endDate, warmupDays)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	
	// Expecting: 2024-01 (Warmup), 2024-02 (Target)
	expectedCount := 2
	if len(urls) != expectedCount {
		t.Errorf("Expected %d urls, got %d", expectedCount, len(urls))
	}
	
	// Check content of first URL (Warmup)
	expectedFirst := expectedBase + "/BTCUSDT/BTCUSDT-trades-2024-01.zip"
	if urls[0] != expectedFirst {
		t.Errorf("Expected warmup URL %s, got %s", expectedFirst, urls[0])
	}
	
	// Check content of second URL (Target)
	expectedSecond := expectedBase + "/BTCUSDT/BTCUSDT-trades-2024-02.zip"
	if urls[1] != expectedSecond {
		t.Errorf("Expected target URL %s, got %s", expectedSecond, urls[1])
	}
}
