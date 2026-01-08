package backtest

import "math"

// CostModel implements the Square Root Market Impact Model.
//
// Impact = η × σ × √(Q / V)
//
// Where:
//   - η: Impact coefficient
//   - σ: Volatility
//   - Q: Order size
//   - V: Average volume
type CostModel struct {
	BaseFee      float64 // Base exchange fee (e.g., 0.001 = 0.1%)
	BaseSlippage float64 // Minimum slippage (e.g., 0.0001 = 0.01%)
	ImpactCoeff  float64 // Market impact coefficient (η)
	SlippageCap  float64 // Maximum slippage cap
}

// NewCostModel creates a new cost model from config.
func NewCostModel(cfg Config) *CostModel {
	return &CostModel{
		BaseFee:      cfg.BaseFee,
		BaseSlippage: cfg.BaseSlippage,
		ImpactCoeff:  cfg.ImpactCoeff,
		SlippageCap:  cfg.SlippageCap,
	}
}

// DefaultCostModel returns a cost model with sensible defaults.
func DefaultCostModel() *CostModel {
	return &CostModel{
		BaseFee:      0.001,  // 0.10% exchange fee
		BaseSlippage: 0.0001, // 0.01% minimum slippage
		ImpactCoeff:  0.1,    // Impact coefficient
		SlippageCap:  0.005,  // 0.50% maximum slippage
	}
}

// GetSlippage calculates slippage using the Square Root Law.
//
// slippage = base_slippage + η × σ × √(trade_size / avg_volume)
//
// The result is capped at SlippageCap.
func (c *CostModel) GetSlippage(volatility, tradeSize, avgVolume float64) float64 {
	if avgVolume <= 0 {
		return c.SlippageCap
	}

	volumeRatio := tradeSize / avgVolume
	if volumeRatio < 0 {
		volumeRatio = 0
	}

	impact := c.ImpactCoeff * volatility * math.Sqrt(volumeRatio)
	totalSlippage := c.BaseSlippage + impact

	// Cap the slippage
	if totalSlippage > c.SlippageCap {
		return c.SlippageCap
	}

	return totalSlippage
}

// GetTotalCost calculates round-trip cost (entry + exit).
//
// Total Cost = 2 × (base_fee + slippage)
func (c *CostModel) GetTotalCost(volatility, tradeSize, avgVolume float64) float64 {
	slippage := c.GetSlippage(volatility, tradeSize, avgVolume)
	return 2 * (c.BaseFee + slippage)
}

// GetEntryCost calculates the cost for entry only.
func (c *CostModel) GetEntryCost(volatility, tradeSize, avgVolume float64) float64 {
	slippage := c.GetSlippage(volatility, tradeSize, avgVolume)
	return c.BaseFee + slippage
}

// ApplyEntrySlippage adjusts the entry price for slippage.
//
// Long: entry_price × (1 + slippage) - pay more
// Short: entry_price × (1 - slippage) - receive less
func (c *CostModel) ApplyEntrySlippage(price float64, direction Direction, volatility, tradeSize, avgVolume float64) float64 {
	slippage := c.GetSlippage(volatility, tradeSize, avgVolume)

	if direction == DirectionLong {
		return price * (1 + slippage)
	}
	// Short
	return price * (1 - slippage)
}

// ApplyExitSlippage adjusts the exit price for slippage.
//
// Long: exit_price × (1 - slippage) - receive less
// Short: exit_price × (1 + slippage) - pay more to cover
func (c *CostModel) ApplyExitSlippage(price float64, direction Direction, volatility, tradeSize, avgVolume float64) float64 {
	slippage := c.GetSlippage(volatility, tradeSize, avgVolume)

	if direction == DirectionLong {
		return price * (1 - slippage)
	}
	// Short
	return price * (1 + slippage)
}

// CalculatePnL computes the P&L for a trade.
//
// For Long: (exit - entry) / entry
// For Short: (entry - exit) / entry
//
// Returns both gross (before fees) and net (after fees) P&L.
func (c *CostModel) CalculatePnL(entryPrice, exitPrice float64, direction Direction) (grossPnL, netPnL float64) {
	if direction == DirectionLong {
		grossPnL = (exitPrice - entryPrice) / entryPrice
	} else {
		grossPnL = (entryPrice - exitPrice) / entryPrice
	}

	// Subtract fees (applied to both entry and exit)
	netPnL = grossPnL - 2*c.BaseFee

	return grossPnL, netPnL
}

// IsViable checks if a trade is likely profitable after costs.
//
// Expected Profit = PT × σ
// Required = Total Cost × Buffer
//
// Returns true if expected profit > required.
func (c *CostModel) IsViable(ptMult, volatility, tradeSize, avgVolume, buffer float64) bool {
	expectedProfit := ptMult * volatility
	totalCost := c.GetTotalCost(volatility, tradeSize, avgVolume)
	required := totalCost * buffer

	return expectedProfit > required
}
