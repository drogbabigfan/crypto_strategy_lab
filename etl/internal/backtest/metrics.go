package backtest

import (
	"math"
)

// MetricsCalculator computes performance metrics from trades.
type MetricsCalculator struct {
	initialCapital float64
	riskPerTrade   float64
	compounding    bool // true: compound returns, false: simple interest
}

// NewMetricsCalculator creates a new metrics calculator.
func NewMetricsCalculator(initialCapital, riskPerTrade float64, compounding bool) *MetricsCalculator {
	return &MetricsCalculator{
		initialCapital: initialCapital,
		riskPerTrade:   riskPerTrade,
		compounding:    compounding,
	}
}

// Calculate computes all metrics from trades and builds equity curve.
// bars parameter is used for mark-to-market equity curve calculation.
func (m *MetricsCalculator) Calculate(trades []Trade, bars []Bar) Result {
	numBars := len(bars)
	if len(trades) == 0 {
		return Result{
			TotalTrades: 0,
			EquityCurve: m.buildFlatEquityCurve(numBars),
		}
	}

	// Build equity curve with mark-to-market valuation
	equityCurve := m.buildEquityCurve(trades, bars)

	// Calculate basic metrics
	totalPnL := 0.0
	grossProfit := 0.0
	grossLoss := 0.0
	wins := 0
	totalHoldBars := 0
	tpCount := 0
	slCount := 0
	timeoutCount := 0
	pnls := make([]float64, len(trades))

	for i, trade := range trades {
		pnls[i] = trade.PnL
		totalPnL += trade.PnL
		totalHoldBars += trade.HoldingBars

		if trade.PnL > 0 {
			wins++
			grossProfit += trade.PnL
		} else {
			grossLoss += trade.PnL // Already negative
		}

		switch trade.ExitReason {
		case ExitReasonTP:
			tpCount++
		case ExitReasonSL:
			slCount++
		case ExitReasonTimeout:
			timeoutCount++
		}
	}

	n := len(trades)
	winRate := float64(wins) / float64(n)
	avgPnL := totalPnL / float64(n)
	avgHoldBars := float64(totalHoldBars) / float64(n)

	// Profit factor
	profitFactor := 0.0
	if grossLoss < 0 {
		profitFactor = grossProfit / (-grossLoss)
	} else if grossProfit > 0 {
		profitFactor = math.Inf(1) // No losses
	}

	// Sharpe ratio from daily returns (annualized with sqrt(252))
	sharpe := m.calculateSharpe(equityCurve, bars)

	// Maximum drawdown
	maxDD := m.calculateMaxDrawdown(equityCurve)

	// Total PnL from equity curve (consistent with MDD/Sharpe)
	equityBasedPnL := 0.0
	if len(equityCurve) > 0 && m.initialCapital > 0 {
		finalEquity := equityCurve[len(equityCurve)-1]
		equityBasedPnL = (finalEquity - m.initialCapital) / m.initialCapital
	}

	return Result{
		TotalTrades:  n,
		WinRate:      winRate,
		AvgPnL:       avgPnL,
		TotalPnL:     equityBasedPnL,
		SharpeRatio:  sharpe,
		MaxDrawdown:  maxDD,
		ProfitFactor: profitFactor,
		AvgHoldBars:  avgHoldBars,
		TPCount:      tpCount,
		SLCount:      slCount,
		TimeoutCount: timeoutCount,
		Trades:       trades,
		EquityCurve:  equityCurve,
	}
}

// buildEquityCurve creates the equity curve from trades with mark-to-market valuation.
// Supports two modes:
// - Compounding (m.compounding=true): reinvest profits, each trade uses current capital
// - Simple interest (m.compounding=false): fixed position size, additive returns
// Uses trade.Size as a multiplier for position sizing (1.0 = 100%, 1.5 = 150%, etc.)
//
// IMPORTANT: Includes liquidation check - if equity drops to 0 or below, the account
// is considered liquidated and equity stays at 0 for the rest of the backtest.
func (m *MetricsCalculator) buildEquityCurve(trades []Trade, bars []Bar) []float64 {
	numBars := len(bars)
	if numBars == 0 {
		return []float64{}
	}

	curve := make([]float64, numBars)
	realizedCapital := m.initialCapital // Capital after all realized trades
	liquidated := false                 // Track if account was liquidated

	// For simple interest mode, use base position size
	basePositionSize := m.initialCapital * m.riskPerTrade

	// For each bar, we need to know: which trade is active (if any)
	tradeIdx := 0

	for barIdx := 0; barIdx < numBars; barIdx++ {
		// If already liquidated, equity stays at 0
		if liquidated {
			curve[barIdx] = 0
			continue
		}

		// Check if we've moved past the current trade's exit
		for tradeIdx < len(trades) && trades[tradeIdx].ExitBar < barIdx {
			// Apply realized P&L with trade's size
			trade := trades[tradeIdx]
			tradeSize := trade.Size
			if tradeSize <= 0 {
				tradeSize = 1.0 // Default to 100% if not set
			}
			if m.compounding {
				realizedCapital *= (1 + trade.PnL*m.riskPerTrade*tradeSize)
			} else {
				realizedCapital += basePositionSize * trade.PnL * tradeSize
			}

			// Check for liquidation after realized P&L
			if realizedCapital <= 0 {
				liquidated = true
				realizedCapital = 0
				curve[barIdx] = 0
				break
			}
			tradeIdx++
		}

		if liquidated {
			continue
		}

		// Check if current bar is within an active trade
		if tradeIdx < len(trades) {
			trade := trades[tradeIdx]
			tradeSize := trade.Size
			if tradeSize <= 0 {
				tradeSize = 1.0 // Default to 100% if not set
			}

			// Check if we're at the exit bar - apply realized P&L
			if barIdx == trade.ExitBar {
				if m.compounding {
					realizedCapital *= (1 + trade.PnL*m.riskPerTrade*tradeSize)
				} else {
					realizedCapital += basePositionSize * trade.PnL * tradeSize
				}

				// Check for liquidation
				if realizedCapital <= 0 {
					liquidated = true
					realizedCapital = 0
				}
				curve[barIdx] = realizedCapital
				tradeIdx++
				continue
			}

			// Check if we're within the trade (after entry, before exit)
			if barIdx >= trade.EntryBar && barIdx < trade.ExitBar {
				// Calculate unrealized P&L using current bar's close price
				currentPrice := bars[barIdx].Close
				var unrealizedPnL float64

				if trade.Direction == DirectionLong {
					unrealizedPnL = (currentPrice - trade.EntryPrice) / trade.EntryPrice
				} else { // Short
					unrealizedPnL = (trade.EntryPrice - currentPrice) / trade.EntryPrice
				}

				// Apply unrealized P&L to equity (mark-to-market) with trade's size
				var currentEquity float64
				if m.compounding {
					currentEquity = realizedCapital * (1 + unrealizedPnL*m.riskPerTrade*tradeSize)
				} else {
					currentEquity = realizedCapital + basePositionSize*unrealizedPnL*tradeSize
				}

				// Check for liquidation on unrealized loss
				if currentEquity <= 0 {
					liquidated = true
					curve[barIdx] = 0
					continue
				}

				curve[barIdx] = currentEquity
				continue
			}
		}

		// No active position - use realized capital
		curve[barIdx] = realizedCapital
	}

	return curve
}

// buildFlatEquityCurve creates a flat equity curve (no trades).
func (m *MetricsCalculator) buildFlatEquityCurve(numBars int) []float64 {
	curve := make([]float64, numBars)
	for i := range curve {
		curve[i] = m.initialCapital
	}
	return curve
}

// calculateSharpe computes the annualized Sharpe ratio from equity curve.
// Uses daily returns with sqrt(252) annualization factor.
func (m *MetricsCalculator) calculateSharpe(equityCurve []float64, bars []Bar) float64 {
	if len(equityCurve) < 2 || len(bars) < 2 {
		return 0
	}

	// Group equity by day and calculate daily returns
	dailyReturns := m.calculateDailyReturns(equityCurve, bars)

	if len(dailyReturns) < 2 {
		return 0
	}

	// Calculate mean daily return
	mean := 0.0
	for _, ret := range dailyReturns {
		mean += ret
	}
	mean /= float64(len(dailyReturns))

	// Calculate standard deviation
	variance := 0.0
	for _, ret := range dailyReturns {
		diff := ret - mean
		variance += diff * diff
	}
	variance /= float64(len(dailyReturns))

	std := math.Sqrt(variance)
	if std < 1e-10 {
		if mean > 0 {
			return math.Inf(1)
		}
		return 0
	}

	// Annualize: sqrt(252 trading days)
	annualizationFactor := math.Sqrt(252)

	return (mean / std) * annualizationFactor
}

// calculateDailyReturns aggregates bar-level equity to daily returns.
func (m *MetricsCalculator) calculateDailyReturns(equityCurve []float64, bars []Bar) []float64 {
	if len(equityCurve) == 0 || len(bars) == 0 {
		return nil
	}

	// milliseconds per day
	const msPerDay int64 = 24 * 60 * 60 * 1000

	// Group by day and get end-of-day equity
	dailyEquity := make(map[int64]float64) // day -> last equity of that day
	var days []int64

	for i, bar := range bars {
		if i >= len(equityCurve) {
			break
		}
		day := bar.Timestamp / msPerDay
		if _, exists := dailyEquity[day]; !exists {
			days = append(days, day)
		}
		dailyEquity[day] = equityCurve[i] // Last bar of the day wins
	}

	if len(days) < 2 {
		return nil
	}

	// Calculate daily returns
	returns := make([]float64, len(days)-1)
	for i := 1; i < len(days); i++ {
		prevEquity := dailyEquity[days[i-1]]
		currEquity := dailyEquity[days[i]]
		if prevEquity > 0 {
			returns[i-1] = (currEquity - prevEquity) / prevEquity
		}
	}

	return returns
}

// calculateMaxDrawdown computes the maximum drawdown from equity curve.
func (m *MetricsCalculator) calculateMaxDrawdown(curve []float64) float64 {
	if len(curve) == 0 {
		return 0
	}

	peak := curve[0]
	maxDD := 0.0

	for _, equity := range curve {
		if equity > peak {
			peak = equity
		}
		if peak > 0 {
			dd := (peak - equity) / peak
			if dd > maxDD {
				maxDD = dd
			}
		}
	}

	return maxDD
}

// CalculateRollingMetrics computes metrics over a rolling window.
// Note: This calculates metrics without mark-to-market equity curve.
func (m *MetricsCalculator) CalculateRollingMetrics(trades []Trade, windowSize int) []Result {
	if len(trades) < windowSize {
		return nil
	}

	results := make([]Result, len(trades)-windowSize+1)
	for i := range results {
		windowTrades := trades[i : i+windowSize]
		results[i] = m.Calculate(windowTrades, []Bar{}) // No equity curve for rolling
	}

	return results
}

// Sortino calculates the Sortino ratio (downside deviation only) from daily returns.
func Sortino(dailyReturns []float64) float64 {
	if len(dailyReturns) == 0 {
		return 0
	}

	mean := 0.0
	for _, ret := range dailyReturns {
		mean += ret
	}
	mean /= float64(len(dailyReturns))

	// Downside deviation (only negative returns)
	downsideVar := 0.0
	for _, ret := range dailyReturns {
		if ret < 0 {
			downsideVar += ret * ret
		}
	}
	downsideVar /= float64(len(dailyReturns))

	downsideStd := math.Sqrt(downsideVar)
	if downsideStd < 1e-10 {
		if mean > 0 {
			return math.Inf(1)
		}
		return 0
	}

	// Annualize: sqrt(252 trading days)
	annualizationFactor := math.Sqrt(252)
	return (mean / downsideStd) * annualizationFactor
}

// Calmar calculates the Calmar ratio (return / max drawdown).
func Calmar(totalReturn, maxDrawdown float64) float64 {
	if maxDrawdown < 1e-10 {
		if totalReturn > 0 {
			return math.Inf(1)
		}
		return 0
	}
	return totalReturn / maxDrawdown
}

// ConsecutiveLosses calculates the maximum consecutive losing trades.
func ConsecutiveLosses(trades []Trade) int {
	maxConsec := 0
	current := 0

	for _, trade := range trades {
		if trade.PnL < 0 {
			current++
			if current > maxConsec {
				maxConsec = current
			}
		} else {
			current = 0
		}
	}

	return maxConsec
}

// ConsecutiveWins calculates the maximum consecutive winning trades.
func ConsecutiveWins(trades []Trade) int {
	maxConsec := 0
	current := 0

	for _, trade := range trades {
		if trade.PnL > 0 {
			current++
			if current > maxConsec {
				maxConsec = current
			}
		} else {
			current = 0
		}
	}

	return maxConsec
}

// ExpectancyPerBar calculates expected return per bar held.
func ExpectancyPerBar(trades []Trade) float64 {
	if len(trades) == 0 {
		return 0
	}

	totalPnL := 0.0
	totalBars := 0
	for _, trade := range trades {
		totalPnL += trade.PnL
		totalBars += trade.HoldingBars
	}

	if totalBars == 0 {
		return 0
	}

	return totalPnL / float64(totalBars)
}
