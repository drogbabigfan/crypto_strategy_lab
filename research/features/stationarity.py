import numpy as np
import pandas as pd

class StationarityEngine:
    def __init__(self):
        pass
    
    def get_weights_ffd(self, d, thres, limit=10000):
        """
        Calculate weights for Fractional Differentiation (Fixed Window).
        Source: Lopez de Prado, Advances in Financial Machine Learning.
        """
        w = [1.0]
        k = 1
        while True:
            w_k = -w[-1] / k * (d - k + 1)
            if abs(w_k) < thres:
                break
            w.append(w_k)
            k += 1
            if k >= limit:
                break
        w = np.array(w[::-1]) # Reverse to apply as dot product? 
        # Usually convolution applies w[0]*x[t] + w[1]*x[t-1]...
        # If w is generated as w0=1, w1=-d... then w corresponds to lag process.
        # So we want w applied to [x[t], x[t-1], ...]
        return w

    def frac_diff_ffd(self, series, d=0.4, thres=1e-5):
        """
        Apply Fractional Differentiation (Fixed Window) to a Series.
        """
        # 1. Get Weights
        w = self.get_weights_ffd(d, thres)
        width = len(w) - 1
        
        # 2. Apply weights via rolling dot product
        # Pandas rolling apply is slow.
        # Faster: signal.convolve or simple loop with numpy strides.
        # But for maintenance, let's use pandas straightforward approach or optimized numpy.
        
        # Optimization:
        # result[i] = sum(w * series[i-width:i+1])
        
        # We can use pd.Series.rolling(window=len(w)).apply(...)
        # But this is very slow. 
        # Let's use simple numpy convolution if series fits in memory.
        
        # w is [w_k, w_{k-1}, ... w_0]?
        # Logic: (1-L)^d x_t = \sum_{k=0} w_k x_{t-k}
        # w generated above: w[0] = 1 (k=0), w[1] = -d (k=1)..
        # So we need to convolve series with w.
        # If we use np.convolve(series, w, mode='valid'), it flips the second array?
        # np.convolve(a, v, mode='valid')
        # output[n] = sum(a[n+k] * v[k])? No.
        # output[n] = (a * v)[n] convolution definition usually flips v.
        # So if we want sum(w[k] * x[t-k]), we should pass w directly if convolve flips it?
        # Let's assume standard filter logic. 
        
        # Current w is [1, -d, ...] (k=0, 1...)
        # We need sum(w[k] * x[t-k]).
        # If we flip w -> [w_k, ... 1]
        # And convolve x with flipped w.
        
        values = series.values
        if len(values) < width + 1:
            return pd.Series(np.nan, index=series.index)
            
        w_flipped = w[::-1] 
        # Actually our get_weights returned w[::-1] already?
        # Re-check get_weights logic.
        # Created w = [1, w1, w2...]
        # Returned w[::-1] -> [wk, ... w1, 1]
        # This [wk...1] applied to [x_{t-k}... x_t] via dot product matches sum(w_k x_{t-k}).
        # So we can just use np.convolve(values, w_from_function, mode='valid')?
        # np.convolve(f, g)[n] = sum(f[m]g[n-m])
        # If valid mode:
        # result alignment is tricky.
        
        # Easiest way in pandas to ensure index alignment:
        output = pd.Series(np.nan, index=series.index)
        
        # Use vectorized stride tricks or just loop for now if window is small?
        # Actually simplest:
        # Use the provided w (which is [wk...1])
        # And rolling dot product.
        
        # w (from func) = [w_k, w_{k-1}, ..., w_0]
        # We want result[t] = w_0*x_t + w_1*x_{t-1} ...
        # Result of rolling(window=len(w)).apply(lambda x: np.dot(x, w))
        # This matches x as [x_{t-k}...x_t] and w as [wk...w0].
        
        # Since w is constant, we can speed this up.
        # But for now, let's use the rolling apply for correctness.
        # Wait, apply is slow.
        # Let's use straightforward numpy loop if needed, or composite via dict.
        
        # DataFrame implementation (fastest pandas native)
        # df_rolled = concat [shift(k) for k in len(w)]
        # This is memory intensive.
        
        # Let's stick to simple Series.rolling (it's slow but safe for now, can optimize later)
        # Or better:
        # w is [w_last ... w_0]
        # We want y[t] = sum(w[i] * x[t - (len-1 - i)])
        
        # Let's revert get_weights to standard [w0, w1...]
        w_standard = self.get_weights_ffd(d, thres)[::-1] # Now [1, -d, ...]
        
        # Weights dict
        weights = {i: w_standard[i] for i in range(len(w_standard))}
        
        # Vectorized sum:
        # result = sum(w_k * series.shift(k))
        # This is reasonably fast in pandas/numpy for k < 1000.
        
        res = 0
        for k, weight in weights.items():
            res += weight * series.shift(k)
            
        # First k elements are NaN
        return res

    def detrend_log_price(self, series, window=100):
        """
        Calculate Detrended Log Price (Context Input).
        Logic: Log(P) - EMA(Log(P)).
        """
        log_p = np.log(series)
        trend = log_p.ewm(span=window).mean()
        return log_p - trend
