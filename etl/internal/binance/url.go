package binance

import (
	"fmt"
	"time"
)

// Market type constants
type MarketType string

const (
	MarketSpot    MarketType = "spot"
	MarketFutures MarketType = "futures"
)

// Base URLs for Binance Vision historical data
const (
	BaseURLSpot    = "https://data.binance.vision/data/spot/monthly/trades"
	BaseURLFutures = "https://data.binance.vision/data/futures/um/monthly/trades"
)

// Data availability start dates
var (
	SpotStartDate    = time.Date(2017, 8, 1, 0, 0, 0, 0, time.UTC)
	FuturesStartDate = time.Date(2019, 9, 1, 0, 0, 0, 0, time.UTC)
)

// GetBaseURL returns the base URL for the given market type
func GetBaseURL(market MarketType) string {
	switch market {
	case MarketFutures:
		return BaseURLFutures
	default:
		return BaseURLSpot
	}
}

// GetStartDate returns the earliest available data date for the given market type
func GetStartDate(market MarketType) time.Time {
	switch market {
	case MarketFutures:
		return FuturesStartDate
	default:
		return SpotStartDate
	}
}

// Deprecated: Use GenerateMonthlyURLsForMarket instead
const BaseURL = BaseURLSpot

// GenerateMonthlyURLsForMarket generates a list of URLs for monthly raw trade data zips.
// It automatically subtracts warmupDays from startDate to include warmup data.
func GenerateMonthlyURLsForMarket(market MarketType, symbol string, startDate, endDate time.Time, warmupDays int) ([]string, error) {
	baseURL := GetBaseURL(market)
	earliestDate := GetStartDate(market)

	// Apply warmup logic
	realStartDate := startDate.AddDate(0, 0, -warmupDays)

	// Clamp to earliest available date
	if realStartDate.Before(earliestDate) {
		realStartDate = earliestDate
	}

	var urls []string
	current := time.Date(realStartDate.Year(), realStartDate.Month(), 1, 0, 0, 0, 0, time.UTC)

	for {
		// If the start of the current month is >= EndDate, then this month is outside the range [..., End).
		if !current.Before(endDate) {
			break
		}

		monthStr := fmt.Sprintf("%02d", current.Month())
		yearStr := fmt.Sprintf("%d", current.Year())
		url := fmt.Sprintf("%s/%s/%s-trades-%s-%s.zip", baseURL, symbol, symbol, yearStr, monthStr)
		urls = append(urls, url)

		current = current.AddDate(0, 1, 0)
	}

	return urls, nil
}

// GenerateMonthlyURLs generates URLs for spot market (backward compatibility)
func GenerateMonthlyURLs(symbol string, startDate, endDate time.Time, warmupDays int) ([]string, error) {
	return GenerateMonthlyURLsForMarket(MarketSpot, symbol, startDate, endDate, warmupDays)
}
