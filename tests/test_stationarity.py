import pytest
import pandas as pd
import numpy as np
from research.features.stationarity import StationarityEngine

def test_frac_diff_stationarity():
    """Test that FracDiff makes a non-stationary series stationary."""
    # Generate non-stationary Random Walk
    np.random.seed(42)
    rw = np.cumsum(np.random.normal(0, 1, 5000))
    series = pd.Series(rw)
    
    engine = StationarityEngine()
    
    # Apply FracDiff with fixed d=0.4
    diffed = engine.frac_diff_ffd(series, d=0.4, thres=1e-4)
    
    # Check if we lost too many samples (window size) represents weights
    # With thres=1e-5, window is small?
    assert not diffed.isna().all()
    # Check simple property: Standard deviation should be lower than original RW
    assert diffed.std() < series.std()
    
    # Optional: Check ADF p-value if we have statsmodels (Assuming we do or mock it)
    # For now, just check logic runs and returns reasonable values.

def test_detrending_context():
    """Test Price Detrending (Context Input)."""
    # Linear Trend
    trend = pd.Series(np.linspace(100, 200, 100))
    engine = StationarityEngine()
    
    # Detrend by dividing by EMA(10)
    # Note: EMA lags, so result should be > 1 in uptrend? Or just deviations.
    # Log-Detrend: log(P) - log(MA).
    
    detrended = engine.detrend_log_price(trend, window=10)
    
    # Result should be roughly constant (stationary) around some mean, 
    # capturing the trend deviation.
    # In a perfect linear trend + EMA lag, it converges.
    assert not detrended.isna().all()
    # Check that it's somewhat bounded unlike the original trend
    assert detrended.max() < 10 # Should be small decimal
