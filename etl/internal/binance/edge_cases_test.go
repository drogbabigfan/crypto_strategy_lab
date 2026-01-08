package binance

import (
	"testing"
	"time"
)

// ============================================================================
// RISK: Trade Parsing Edge Cases
// ============================================================================

func TestParseTradeEmptyFields(t *testing.T) {
	// Empty strings
	record := []string{"", "", "", "", "", "", ""}

	_, err := ParseTrade(record)
	if err == nil {
		t.Error("Should error on empty fields")
	}
}

func TestParseTradeMissingFields(t *testing.T) {
	// Too few fields
	record := []string{"12345", "50000.00", "1.5"}

	_, err := ParseTrade(record)
	if err == nil {
		t.Error("Should error on missing fields")
	}
}

func TestParseTradeInvalidPrice(t *testing.T) {
	record := []string{"12345", "not_a_number", "1.5", "75000", "1600000000000", "true", "true"}

	_, err := ParseTrade(record)
	if err == nil {
		t.Error("Should error on invalid price")
	}
}

func TestParseTradeInvalidQuantity(t *testing.T) {
	record := []string{"12345", "50000.00", "invalid", "75000", "1600000000000", "true", "true"}

	_, err := ParseTrade(record)
	if err == nil {
		t.Error("Should error on invalid quantity")
	}
}

func TestParseTradeInvalidTimestamp(t *testing.T) {
	record := []string{"12345", "50000.00", "1.5", "75000", "not_a_timestamp", "true", "true"}

	_, err := ParseTrade(record)
	if err == nil {
		t.Error("Should error on invalid timestamp")
	}
}

func TestParseTradeExtremeValues(t *testing.T) {
	// Very large values
	record := []string{
		"9999999999999",    // Large ID
		"999999999.99999",  // Large price
		"99999999.99999",   // Large quantity
		"9999999999999.99", // Large quote qty
		"9999999999999",    // Large timestamp
		"false",
		"true",
	}

	trade, err := ParseTrade(record)
	if err != nil {
		t.Fatalf("Should handle large values: %v", err)
	}

	if trade.ID != 9999999999999 {
		t.Errorf("Expected ID 9999999999999, got %d", trade.ID)
	}
}

func TestParseTradeScientificNotation(t *testing.T) {
	// Scientific notation (sometimes appears in data)
	record := []string{"12345", "5.0e4", "1.5e-2", "750", "1600000000000", "true", "true"}

	trade, err := ParseTrade(record)
	if err != nil {
		t.Fatalf("Should handle scientific notation: %v", err)
	}

	if trade.Price != 50000 {
		t.Errorf("Expected price 50000, got %f", trade.Price)
	}

	if trade.Quantity != 0.015 {
		t.Errorf("Expected quantity 0.015, got %f", trade.Quantity)
	}
}

func TestParseTradeWhitespace(t *testing.T) {
	// Whitespace around values - should now be trimmed and parsed correctly
	record := []string{" 12345 ", " 50000.00 ", " 1.5 ", " 75000 ", " 1600000000000 ", " true ", " true "}

	trade, err := ParseTrade(record)
	if err != nil {
		t.Fatalf("Should handle whitespace: %v", err)
	}

	if trade.ID != 12345 {
		t.Errorf("Expected ID 12345, got %d", trade.ID)
	}
	if trade.Price != 50000.00 {
		t.Errorf("Expected price 50000, got %f", trade.Price)
	}

	t.Log("Whitespace correctly trimmed and parsed")
}

// ============================================================================
// RISK: Validation Edge Cases
// ============================================================================

func TestParseTradeNegativePrice(t *testing.T) {
	record := []string{"12345", "-50000.00", "1.5", "75000", "1600000000000", "true", "true"}

	_, err := ParseTrade(record)
	if err == nil {
		t.Fatal("Should reject negative price")
	}

	t.Logf("Negative price correctly rejected: %v", err)
}

func TestParseTradeNegativeQuantity(t *testing.T) {
	record := []string{"12345", "50000.00", "-1.5", "-75000", "1600000000000", "true", "true"}

	_, err := ParseTrade(record)
	if err == nil {
		t.Fatal("Should reject negative quantity")
	}

	t.Logf("Negative quantity correctly rejected: %v", err)
}

func TestParseTradeZeroTimestamp(t *testing.T) {
	record := []string{"12345", "50000.00", "1.5", "75000", "0", "true", "true"}

	_, err := ParseTrade(record)
	if err == nil {
		t.Fatal("Should reject zero timestamp")
	}

	t.Logf("Zero timestamp correctly rejected: %v", err)
}

func TestParseTradeZeroID(t *testing.T) {
	record := []string{"0", "50000.00", "1.5", "75000", "1600000000000", "true", "true"}

	_, err := ParseTrade(record)
	if err == nil {
		t.Fatal("Should reject zero trade ID")
	}

	t.Logf("Zero trade ID correctly rejected: %v", err)
}

// ============================================================================
// RISK: URL Generation Edge Cases
// ============================================================================

func TestGenerateMonthlyURLsEmptyRange(t *testing.T) {
	// End before start
	startDate := time.Date(2021, 6, 1, 0, 0, 0, 0, time.UTC)
	endDate := time.Date(2021, 5, 1, 0, 0, 0, 0, time.UTC)

	urls, err := GenerateMonthlyURLs("BTCUSDT", startDate, endDate, 0)
	if err != nil {
		t.Logf("Empty range error (expected): %v", err)
	}

	if len(urls) != 0 {
		t.Errorf("Expected empty slice for invalid range, got %d URLs", len(urls))
	}
}

func TestGenerateMonthlyURLsSameMonth(t *testing.T) {
	startDate := time.Date(2021, 6, 1, 0, 0, 0, 0, time.UTC)
	endDate := time.Date(2021, 6, 30, 0, 0, 0, 0, time.UTC)

	urls, err := GenerateMonthlyURLs("BTCUSDT", startDate, endDate, 0)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	if len(urls) != 1 {
		t.Errorf("Expected 1 URL for same month, got %d", len(urls))
	}
}

func TestGenerateMonthlyURLsYearBoundary(t *testing.T) {
	// November 2020 to February 2021
	startDate := time.Date(2020, 11, 1, 0, 0, 0, 0, time.UTC)
	endDate := time.Date(2021, 2, 28, 0, 0, 0, 0, time.UTC)

	urls, err := GenerateMonthlyURLs("BTCUSDT", startDate, endDate, 0)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	if len(urls) != 4 {
		t.Errorf("Expected 4 URLs across year boundary, got %d", len(urls))
	}

	// Verify months are correct
	expectedMonths := []string{"2020-11", "2020-12", "2021-01", "2021-02"}
	for i, url := range urls {
		found := false
		for _, month := range expectedMonths {
			if contains(url, month) {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("URL %d doesn't contain expected month: %s", i, url)
		}
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > 0 && containsHelper(s, substr))
}

func containsHelper(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
