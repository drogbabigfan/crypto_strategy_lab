import pytest
import pandas as pd
import numpy as np
from research.labels.barrier import TripleBarrierLabeler

def test_triple_barrier_hit_pt():
    """Test identifying a Take Profit hit."""
    # Price path: 100 -> 101 -> 102 (Target met)
    prices = pd.Series([100, 101, 102], index=pd.to_datetime(["2024-01-01 10:00", "2024-01-01 11:00", "2024-01-01 12:00"]))
    volatility = pd.Series([0.01, 0.01, 0.01], index=prices.index) # Sigma=1%

    # PT = 1.0 * Sigma = 0.01. Target = 101.
    labeler = TripleBarrierLabeler(pt_mult=1.0, sl_mult=1.0, vertical_hours=10)
    
    # Logic: For bar 0 (100):
    # Horizon is up to end.
    # Highs: 101 (t1), 102 (t2).
    # 101 >= 100 + 1.0? Yes. First hit at t1.
    # Label should be 1 (Long).
    
    labels = labeler.label(prices, volatility)
    
    # Expected: First bar labeled 1.
    assert labels.iloc[0]["label"] == 1
    assert labels.iloc[0]["ret"] > 0

def test_triple_barrier_hit_sl():
    """Test identifying a Stop Loss hit."""
    # Price path: 100 -> 99 -> 98
    prices = pd.Series([100, 99, 98], index=pd.to_datetime(["2024-01-01 10:00", "2024-01-01 11:00", "2024-01-01 12:00"]))
    volatility = pd.Series([0.01, 0.01, 0.01], index=prices.index)
    
    # SL = 1.0 * Sigma = 0.01. Lower Bound = 99.
    labeler = TripleBarrierLabeler(pt_mult=1.0, sl_mult=1.0, vertical_hours=10)
    
    labels = labeler.label(prices, volatility)
    
    # Expected: First bar labeled -1 (Short successful? Or Long failed?)
    # Usually strategy Direction:
    # If we classify "Direction", implies deciding Long/Short/Neutral.
    # Standard Triple Barrier labels "First Touch". 
    # If Top touched first -> 1. If Bottom touched first -> -1. If Vertical -> 0.
    # So if price drops, Bottom touched. Label -1.
    assert labels.iloc[0]["label"] == -1

def test_hybrid_vertical_barrier():
    """Test Hybrid Vertical Barrier: Min Bars OR Max Time."""
    # Scenario: Price flat (0 volatility).
    # Should exit at Hybrid Barrier.
    # Hybrid: Min Bars (e.g. 2) OR Max Time (e.g. 1h).
    
    times = pd.date_range("2024-01-01 10:00", periods=10, freq="15min")
    prices = pd.Series([100]*10, index=times)
    volatility = pd.Series([1.0]*10, index=times)
    
    # Max Time 30min (2 bars). Min Bars 5.
    # Hybrid Logic: Close if (Bars > 5) OR (Time > 30min).
    # Actually usually it's "Wait at least Min Bars, but force close at Max Time".
    # Blueprint says: "Close if Bars > 20 OR Time > 72h".
    # This means if either condition is met (whichever comes first? No, usually OR implies EITHER triggers exit).
    # If I wait 72h but only 1 bar passed? Close.
    # If I have 20 bars in 1 min? Close.
    # So it's min(Time, Bars) effectively in terms of duration?
    # Yes. Whichever comes first.
    
    labeler = TripleBarrierLabeler(pt_mult=10.0, sl_mult=10.0, 
                                   vertical_hours=0.5, # 30 mins
                                   min_bars=5)
    
    # Bar 0 (10:00). Max Time (10:30). Bars (10:00 + 5*15m = 11:15).
    # Exit should be at 10:30 (Index 2).
    
    events = labeler.get_events(prices, volatility)
    # Check t1 (end time) for the first bar.
    t1 = events.iloc[0]["t1"]
    
    expected_exit = times[2] # 10:30
    assert t1 == expected_exit, f"Expected exit {expected_exit}, got {t1}"
