package features

// ConnorsRSI implements a modified Connors RSI indicator using streaming algorithms.
// Modified for Dollar Bars: (RSI(3) + PercentRank(100)) / 2
//
// Original Connors RSI includes a "streak" component, but Dollar Bars have
// inconsistent time intervals (time warping), making streak counts meaningless.
// We remove the streak component and use only:
// 1. RSI(3): 3-period RSI of price
// 2. PercentRank(100): Percentile rank of current return over last 100 bars
type ConnorsRSI struct {
	// RSI(3) components - EWM of gains and losses
	rsiPeriod int
	avgGain   float64
	avgLoss   float64
	rsiCount  int64
	rsiAlpha  float64 // Wilder's smoothing: 1/period

	// Percent Rank - ring buffer for recent returns
	returnBuffer *ringBuffer
	rankPeriod   int

	// Previous values
	prevClose float64

	// Current output
	current float64
	count   int64
}

// ConnorsRSIConfig holds configuration.
type ConnorsRSIConfig struct {
	RSIPeriod  int // Default: 3
	RankPeriod int // Default: 100
}

// DefaultConnorsRSIConfig returns default configuration.
func DefaultConnorsRSIConfig() ConnorsRSIConfig {
	return ConnorsRSIConfig{
		RSIPeriod:  3,
		RankPeriod: 100,
	}
}

// NewConnorsRSI creates a new Connors RSI calculator.
func NewConnorsRSI(config ConnorsRSIConfig) *ConnorsRSI {
	return &ConnorsRSI{
		rsiPeriod:    config.RSIPeriod,
		rsiAlpha:     1.0 / float64(config.RSIPeriod),
		returnBuffer: newRingBuffer(config.RankPeriod),
		rankPeriod:   config.RankPeriod,
	}
}

// Update processes a new bar and returns the updated Connors RSI value.
func (c *ConnorsRSI) Update(close float64) float64 {
	c.count++

	// First bar - just store close
	if c.count == 1 {
		c.prevClose = close
		return 0
	}

	// Calculate return
	ret := 0.0
	if c.prevClose > Epsilon {
		ret = (close - c.prevClose) / c.prevClose
	}

	// === Component 1: RSI(3) ===
	rsi3 := c.updateRSI(ret)

	// === Component 2: Percent Rank ===
	percentRank := c.updatePercentRank(ret)

	// === Combine (without Streak) ===
	// Modified ConnorsRSI = (RSI(3) + PercentRank) / 2
	c.current = (rsi3 + percentRank) / 2.0

	c.prevClose = close
	return c.current
}

// updateRSI computes RSI using Wilder's smoothing method.
func (c *ConnorsRSI) updateRSI(ret float64) float64 {
	c.rsiCount++

	gain := 0.0
	loss := 0.0
	if ret > 0 {
		gain = ret
	} else {
		loss = -ret
	}

	if c.rsiCount == 1 {
		c.avgGain = gain
		c.avgLoss = loss
	} else {
		// Wilder's smoothing: EWM with alpha = 1/period
		c.avgGain = c.rsiAlpha*gain + (1-c.rsiAlpha)*c.avgGain
		c.avgLoss = c.rsiAlpha*loss + (1-c.rsiAlpha)*c.avgLoss
	}

	// RSI = 100 - 100 / (1 + RS)
	// where RS = avgGain / avgLoss
	if c.avgLoss < Epsilon {
		if c.avgGain < Epsilon {
			return 50 // No movement
		}
		return 100 // All gains
	}

	rs := c.avgGain / c.avgLoss
	return 100 - 100/(1+rs)
}

// updatePercentRank computes the percentile rank of the current return.
func (c *ConnorsRSI) updatePercentRank(ret float64) float64 {
	// Add to buffer first (so current value is included in ranking)
	c.returnBuffer.Push(ret)

	values := c.returnBuffer.Values()
	if len(values) < 2 {
		return 50 // Default to middle
	}

	// Count how many values are less than current return
	countLess := 0
	for _, v := range values {
		if v < ret {
			countLess++
		}
	}

	// Percent rank = (count less than current) / (total - 1) * 100
	// We use (n-1) because we exclude the current value from denominator
	return float64(countLess) / float64(len(values)-1) * 100
}

// Current returns the current Connors RSI value.
func (c *ConnorsRSI) Current() float64 {
	return c.current
}

// IsPrimed returns true if enough data has been processed.
func (c *ConnorsRSI) IsPrimed() bool {
	return c.returnBuffer.IsFull()
}

// Reset clears all state.
func (c *ConnorsRSI) Reset() {
	c.avgGain = 0
	c.avgLoss = 0
	c.rsiCount = 0
	c.returnBuffer.Reset()
	c.prevClose = 0
	c.current = 0
	c.count = 0
}

// ConnorsRSIState represents serializable state.
type ConnorsRSIState struct {
	AvgGain      float64
	AvgLoss      float64
	RSICount     int64
	ReturnBuffer []float64
	ReturnIndex  int
	ReturnCount  int
	PrevClose    float64
	Count        int64
}

// State returns serializable state snapshot.
func (c *ConnorsRSI) State() ConnorsRSIState {
	return ConnorsRSIState{
		AvgGain:      c.avgGain,
		AvgLoss:      c.avgLoss,
		RSICount:     c.rsiCount,
		ReturnBuffer: append([]float64{}, c.returnBuffer.data...),
		ReturnIndex:  c.returnBuffer.index,
		ReturnCount:  c.returnBuffer.count,
		PrevClose:    c.prevClose,
		Count:        c.count,
	}
}

// LoadState restores from serialized state.
func (c *ConnorsRSI) LoadState(state ConnorsRSIState) {
	c.avgGain = state.AvgGain
	c.avgLoss = state.AvgLoss
	c.rsiCount = state.RSICount
	copy(c.returnBuffer.data, state.ReturnBuffer)
	c.returnBuffer.index = state.ReturnIndex
	c.returnBuffer.count = state.ReturnCount
	c.prevClose = state.PrevClose
	c.count = state.Count
}
