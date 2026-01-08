import pandas as pd
import numpy as np

class TripleBarrierLabeler:
    def __init__(self, pt_mult=1.0, sl_mult=1.0, vertical_hours=48.0, min_bars=20):
        self.pt_mult = pt_mult
        self.sl_mult = sl_mult
        self.vertical_hours = vertical_hours
        self.min_bars = min_bars
    
    def get_events(self, prices, volatility):
        """
        Find the first touch (Barrier Hit) for each timestamp.
        Returns DataFrame with ['t1', 'type', 'ret'].
        type: 1 (PT), -1 (SL), 0 (Vertical)
        """
        events = []
        index = prices.index
        
        # Pre-compute vertical barriers (Time Based)
        # For each t0, t1_time = t0 + timedelta
        t1_times = index + pd.Timedelta(hours=self.vertical_hours)
        
        for i, t0 in enumerate(index):
            # 1. Determine Vertical Barrier (Hybrid)
            # Option A: Time limit
            t1_time_idx = prices.index.searchsorted(t1_times[i])
            # Handle out of bounds
            if t1_time_idx >= len(prices):
                t1_time_idx = len(prices) - 1
                
            # Option B: Bar limit
            t1_bar_idx = i + self.min_bars
            if t1_bar_idx >= len(prices):
                t1_bar_idx = len(prices) - 1
            
            # Hybrid: Whichever comes FIRST (Standard Logic for OR)
            # Wait. User requirement: "Hybrid Barrier (Min Bars OR Max Time)".
            # "Close if Bars > 20 OR Time > 72h".
            # This means if I reach 20 bars, close. If I reach 72h, close.
            # So the horizon is min(date_at_20_bars, date_at_72h).
            
            end_idx = min(t1_time_idx, t1_bar_idx)
            
            # If end_idx <= i (e.g. at end of series), skip
            if end_idx <= i:
                continue
                
            # 2. Get Path
            path = prices.iloc[i : end_idx+1] # Path from t0 to t1
            
            # 3. Set Barriers
            trgt = volatility.iloc[i]
            # If trgt is NaN or 0, we can't label.
            if pd.isna(trgt) or trgt <= 0:
                continue
                
            start_price = path.iloc[0]
            pt_level = start_price * (1 + self.pt_mult * trgt)
            sl_level = start_price * (1 - self.sl_mult * trgt)
            
            # 4. Check Hits
            # PT hit: price >= pt
            pt_hits = path[path >= pt_level]
            # SL hit: price <= sl
            sl_hits = path[path <= sl_level]
            
            pt_time = pt_hits.index[0] if not pt_hits.empty else pd.NaT
            sl_time = sl_hits.index[0] if not sl_hits.empty else pd.NaT
            
            # Find earliest
            t1 = path.index[-1] # Default Vertical
            side = 0
            
            # Logic to pick earliest of pt_time, sl_time, t1
            # Note: t1 is the vertical barrier. pt/sl must happen BEFORE or AT t1.
            # Since path is sliced up to t1, any hit in path is valid.
            
            candidates = []
            if not pd.isna(pt_time): candidates.append((pt_time, 1))
            if not pd.isna(sl_time): candidates.append((sl_time, -1))
            
            if candidates:
                # Sort by time
                candidates.sort(key=lambda x: x[0])
                first_hit_time, first_hit_type = candidates[0]
                
                # If first hit is before or at t1 (it must be, path is sliced)
                t1 = first_hit_time
                side = first_hit_type
            else:
                # Vertical hit
                side = 0 # Or sign of return?
                # Usually Vertical -> 0 implies "timeout".
                # Sign of return labeling is implicit if we use 'ret' column.
                # But 'label' column usually means "Hit PT" (1) or "Hit SL" (-1) or "Timeout" (0).
                # Strategy might treat 0 as "Exit at Market".
            
            ret = (prices[t1] - start_price) / start_price
            
            events.append({
                "t0": t0,
                "t1": t1,
                "label": side,
                "ret": ret
            })
            
        return pd.DataFrame(events).set_index("t0")

    def label(self, prices, volatility):
        return self.get_events(prices, volatility)
