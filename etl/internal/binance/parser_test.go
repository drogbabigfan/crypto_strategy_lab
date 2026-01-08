package binance

import (
	"encoding/csv"
	"strings"
	"testing"
)

func TestParseTrade(t *testing.T) {
	// Raw Trade Format:
	// id, price, qty, quote_qty, time, is_buyer_maker, is_best_match
	// 12345, 99.5, 0.1, 9.95, 1672531200000, true, true

	record := []string{"12345", "99.5", "0.1", "9.95", "1672531200000", "true", "true"}

	trade, err := ParseTrade(record)
	if err != nil {
		t.Fatalf("ParseTrade failed: %v", err)
	}

	if trade.ID != 12345 {
		t.Errorf("Expected ID 12345, got %d", trade.ID)
	}
	if trade.Price != 99.5 {
		t.Errorf("Expected Price 99.5, got %f", trade.Price)
	}
	if trade.Quantity != 0.1 {
		t.Errorf("Expected Qty 0.1, got %f", trade.Quantity)
	}
	if trade.Time != 1672531200000 {
		t.Errorf("Expected Time 1672531200000, got %d", trade.Time)
	}
	if trade.IsBuyerMaker != true {
		t.Errorf("Expected IsBuyerMaker true, got %v", trade.IsBuyerMaker)
	}
}

func TestStreamParsing(t *testing.T) {
	// Simulate CSV Content
	csvContent := `id,price,qty,quote_qty,time,is_buyer_maker,is_best_match
1,100.0,0.1,10.0,1000,false,true
2,100.1,0.2,20.02,1001,true,true
`
	reader := csv.NewReader(strings.NewReader(csvContent))

	// Skip Header
	_, _ = reader.Read()

	count := 0
	for {
		record, err := reader.Read()
		if err != nil {
			break
		}
		_, err = ParseTrade(record)
		if err != nil {
			t.Fatalf("Error parsing row %d: %v", count, err)
		}
		count++
	}

	if count != 2 {
		t.Errorf("Expected 2 trades, got %d", count)
	}
}
