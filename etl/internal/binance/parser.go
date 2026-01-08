package binance

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

// ErrInvalidTrade is returned when trade data fails validation.
var ErrInvalidTrade = fmt.Errorf("invalid trade data")

type Trade struct {
	ID            int64
	Price         float64
	Quantity      float64
	QuoteQuantity float64 // Dollar value (Price * Quantity)
	Time          int64
	IsBuyerMaker  bool
	IsBestMatch   bool

	// Derived field: true if the aggressor (taker) is a buyer.
	// Logic: If IsBuyerMaker == true, maker is buyer, so taker is seller.
	//        If IsBuyerMaker == false, maker is seller, so taker is buyer.
	IsAggressorBuy bool
}

// ValidateTrade checks if trade values are within valid ranges.
// Returns an error describing the validation failure, or nil if valid.
func ValidateTrade(t *Trade) error {
	// Price must be positive and finite
	if t.Price <= 0 {
		return fmt.Errorf("%w: price must be positive, got %f", ErrInvalidTrade, t.Price)
	}
	if math.IsNaN(t.Price) || math.IsInf(t.Price, 0) {
		return fmt.Errorf("%w: price is NaN or Inf", ErrInvalidTrade)
	}

	// Quantity must be non-negative and finite
	if t.Quantity < 0 {
		return fmt.Errorf("%w: quantity must be non-negative, got %f", ErrInvalidTrade, t.Quantity)
	}
	if math.IsNaN(t.Quantity) || math.IsInf(t.Quantity, 0) {
		return fmt.Errorf("%w: quantity is NaN or Inf", ErrInvalidTrade)
	}

	// QuoteQuantity must be non-negative and finite
	if t.QuoteQuantity < 0 {
		return fmt.Errorf("%w: quote_qty must be non-negative, got %f", ErrInvalidTrade, t.QuoteQuantity)
	}
	if math.IsNaN(t.QuoteQuantity) || math.IsInf(t.QuoteQuantity, 0) {
		return fmt.Errorf("%w: quote_qty is NaN or Inf", ErrInvalidTrade)
	}

	// Timestamp must be positive (after Unix epoch)
	if t.Time <= 0 {
		return fmt.Errorf("%w: timestamp must be positive, got %d", ErrInvalidTrade, t.Time)
	}

	// Trade ID must be positive
	if t.ID <= 0 {
		return fmt.Errorf("%w: trade ID must be positive, got %d", ErrInvalidTrade, t.ID)
	}

	return nil
}

// ParseTrade parses a CSV record into a Trade struct.
// Supports both Spot format (7 fields) and Futures format (6 fields).
// Spot format: id, price, qty, quote_qty, time, is_buyer_maker, is_best_match
// Futures format: id, price, qty, quote_qty, time, is_buyer_maker
// Applies whitespace trimming and validation.
func ParseTrade(record []string) (*Trade, error) {
	// Support both Spot (7 fields) and Futures (6 fields) formats
	if len(record) < 6 {
		return nil, fmt.Errorf("insufficient fields: got %d, expected at least 6", len(record))
	}

	// Trim whitespace from all fields
	for i := range record {
		record[i] = strings.TrimSpace(record[i])
	}

	id, err := strconv.ParseInt(record[0], 10, 64)
	if err != nil {
		return nil, fmt.Errorf("invalid id: %w", err)
	}

	price, err := strconv.ParseFloat(record[1], 64)
	if err != nil {
		return nil, fmt.Errorf("invalid price: %w", err)
	}

	qty, err := strconv.ParseFloat(record[2], 64)
	if err != nil {
		return nil, fmt.Errorf("invalid qty: %w", err)
	}

	quoteQty, err := strconv.ParseFloat(record[3], 64)
	if err != nil {
		return nil, fmt.Errorf("invalid quote_qty: %w", err)
	}

	timeVal, err := strconv.ParseInt(record[4], 10, 64)
	if err != nil {
		return nil, fmt.Errorf("invalid time: %w", err)
	}

	isBuyerMaker, err := strconv.ParseBool(record[5])
	if err != nil {
		return nil, fmt.Errorf("invalid is_buyer_maker: %w", err)
	}

	// is_best_match is optional (only present in Spot data, not in Futures)
	isBestMatch := true // Default to true (same as Spot behavior)
	if len(record) >= 7 && record[6] != "" {
		isBestMatch, err = strconv.ParseBool(record[6])
		if err != nil {
			return nil, fmt.Errorf("invalid is_best_match: %w", err)
		}
	}

	trade := &Trade{
		ID:            id,
		Price:         price,
		Quantity:      qty,
		QuoteQuantity: quoteQty,
		Time:          timeVal,
		IsBuyerMaker:  isBuyerMaker,
		IsBestMatch:   isBestMatch,
		// Derive aggressor side:
		// IsBuyerMaker == true  -> Maker is Buyer  -> Taker is Seller -> IsAggressorBuy = false
		// IsBuyerMaker == false -> Maker is Seller -> Taker is Buyer  -> IsAggressorBuy = true
		IsAggressorBuy: !isBuyerMaker,
	}

	// Validate parsed trade
	if err := ValidateTrade(trade); err != nil {
		return nil, err
	}

	return trade, nil
}
