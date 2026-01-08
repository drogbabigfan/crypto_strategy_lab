package bars

// DynamicDollarBar represents an adaptive dollar bar with microstructure features.
// Designed for Mid-Frequency Trading (MFT) Deep Learning models.
// Raw Data First: No normalization/scaling - preserves original values for model flexibility.
type DynamicDollarBar struct {
	// 1. Basic OHLCV
	StartTime int64   `parquet:"start_time"` // Bar start time (ms)
	EndTime   int64   `parquet:"end_time"`   // Bar end time (ms)
	Open      float64 `parquet:"open"`
	High      float64 `parquet:"high"`
	HighTime  int64   `parquet:"high_time"` // Timestamp when High occurred (ms) - for precise backtest
	Low       float64 `parquet:"low"`
	LowTime   int64   `parquet:"low_time"` // Timestamp when Low occurred (ms) - for precise backtest
	Close     float64 `parquet:"close"`
	Volume    float64 `parquet:"volume"` // Total Quantity

	// 2. Bar Info
	DollarValue   float64 `parquet:"dollar_value"`   // Accumulated Quote Qty (Price * Qty)
	TickCount     int64   `parquet:"tick_count"`
	ThresholdUsed float64 `parquet:"threshold_used"` // The actual D_t used to cut this bar
	Duration      float64 `parquet:"duration"`       // [Time-Aware] Bar formation time in seconds

	// 3. Imbalance Features (Alpha Source)
	// Buy = Aggressor is Buyer (Taker), Sell = Aggressor is Seller (Taker)
	BuyDollarVol  float64 `parquet:"buy_dollar_vol"`
	SellDollarVol float64 `parquet:"sell_dollar_vol"`
	NetImbalance  float64 `parquet:"net_imbalance"` // (BuyDollarVol - SellDollarVol)
}

// BarAccumulator holds the in-progress state for building a bar.
// Designed for zero-allocation accumulation.
type BarAccumulator struct {
	StartTimestamp int64 // First trade timestamp (for bar start time)
	EndTimestamp   int64 // Last trade timestamp (for bar end time)
	MinTimestamp   int64 // Minimum timestamp seen (for accurate duration)
	MaxTimestamp   int64 // Maximum timestamp seen (for accurate duration)
	Open           float64
	High           float64
	HighTimestamp  int64 // Timestamp when High occurred (for precise backtest)
	Low            float64
	LowTimestamp   int64 // Timestamp when Low occurred (for precise backtest)
	Close          float64
	Volume         float64
	DollarValue    float64
	TickCount      int64
	BuyDollarVol   float64
	SellDollarVol  float64
}

// Reset clears the accumulator for the next bar.
func (a *BarAccumulator) Reset() {
	a.StartTimestamp = 0
	a.EndTimestamp = 0
	a.MinTimestamp = 0
	a.MaxTimestamp = 0
	a.Open = 0
	a.High = 0
	a.HighTimestamp = 0
	a.Low = 0
	a.LowTimestamp = 0
	a.Close = 0
	a.Volume = 0
	a.DollarValue = 0
	a.TickCount = 0
	a.BuyDollarVol = 0
	a.SellDollarVol = 0
}

// AddTrade accumulates a single trade into the bar.
// isAggressorBuy: true if taker is buyer, false if taker is seller.
func (a *BarAccumulator) AddTrade(price, quantity, dollarValue float64, timestamp int64, isAggressorBuy bool) {
	// First tick initializes OHLC and timestamps
	if a.TickCount == 0 {
		a.StartTimestamp = timestamp
		a.MinTimestamp = timestamp
		a.MaxTimestamp = timestamp
		a.Open = price
		a.High = price
		a.HighTimestamp = timestamp
		a.Low = price
		a.LowTimestamp = timestamp
	}

	// Update High/Low with timestamps
	if price > a.High {
		a.High = price
		a.HighTimestamp = timestamp
	}
	if price < a.Low {
		a.Low = price
		a.LowTimestamp = timestamp
	}

	// Track Min/Max timestamps for accurate duration calculation
	// (handles out-of-order data gracefully)
	if timestamp < a.MinTimestamp {
		a.MinTimestamp = timestamp
	}
	if timestamp > a.MaxTimestamp {
		a.MaxTimestamp = timestamp
	}

	// Always update Close and EndTimestamp (last processed trade)
	a.Close = price
	a.EndTimestamp = timestamp

	// Accumulate volume and dollar value
	a.Volume += quantity
	a.DollarValue += dollarValue
	a.TickCount++

	// Accumulate imbalance
	if isAggressorBuy {
		a.BuyDollarVol += dollarValue
	} else {
		a.SellDollarVol += dollarValue
	}
}

// MinDuration is the minimum duration in seconds to avoid Division by Zero.
// HFT scenarios can complete bars within the same millisecond (StartTime == EndTime).
// Without this floor, Price/Duration or Log(Duration) calculations would produce
// Division by Zero or -Infinity, crashing model training.
const MinDuration = 0.001 // 1 millisecond floor

// Finalize creates a DynamicDollarBar from the accumulated state.
func (a *BarAccumulator) Finalize(thresholdUsed float64) DynamicDollarBar {
	// Calculate duration in seconds using Min/Max timestamps
	// This handles out-of-order data correctly: duration = (MaxTime - MinTime) / 1000
	// Apply minimum clipping to prevent Division by Zero in downstream calculations
	duration := float64(a.MaxTimestamp-a.MinTimestamp) / 1000.0
	if duration < MinDuration {
		duration = MinDuration
	}

	return DynamicDollarBar{
		StartTime:     a.MinTimestamp, // Use MinTimestamp as bar start (more accurate)
		EndTime:       a.MaxTimestamp, // Use MaxTimestamp as bar end (more accurate)
		Open:          a.Open,
		High:          a.High,
		HighTime:      a.HighTimestamp,
		Low:           a.Low,
		LowTime:       a.LowTimestamp,
		Close:         a.Close,
		Volume:        a.Volume,
		DollarValue:   a.DollarValue,
		TickCount:     a.TickCount,
		ThresholdUsed: thresholdUsed,
		Duration:      duration,
		BuyDollarVol:  a.BuyDollarVol,
		SellDollarVol: a.SellDollarVol,
		NetImbalance:  a.BuyDollarVol - a.SellDollarVol,
	}
}
