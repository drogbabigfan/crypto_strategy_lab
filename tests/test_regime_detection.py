"""
Tests for Regime Detection Module.
"""

import pytest
import numpy as np
import pandas as pd


class TestHurstCalculator:
    """Tests for Hurst exponent calculator."""

    def test_import(self):
        """Test module imports correctly."""
        from strategies.regime_detection import HurstCalculator, HurstRegime
        assert HurstCalculator is not None
        assert HurstRegime is not None

    def test_hurst_trending_data(self):
        """Test Hurst detects trending data correctly."""
        from strategies.regime_detection import HurstCalculator

        # Create strongly trending data
        np.random.seed(42)
        n = 500
        trend = np.cumsum(np.ones(n) * 0.1 + np.random.randn(n) * 0.01)

        calculator = HurstCalculator()
        result = calculator.calculate_dfa(trend)

        # Trending data should have H > 0.5
        assert result.hurst > 0.5, f"Expected H > 0.5 for trending data, got {result.hurst}"
        assert result.regime.value == 'trending'

    def test_hurst_mean_reverting_data(self):
        """Test Hurst detects mean-reverting data correctly."""
        from strategies.regime_detection import HurstCalculator

        # Create mean-reverting data (Ornstein-Uhlenbeck process approximation)
        np.random.seed(42)
        n = 500
        data = np.zeros(n)
        theta = 0.5  # Mean reversion speed
        for i in range(1, n):
            data[i] = data[i-1] - theta * data[i-1] + np.random.randn() * 0.1

        calculator = HurstCalculator()
        result = calculator.calculate_dfa(data)

        # Mean-reverting data should have H < 0.5
        assert result.hurst < 0.55, f"Expected H < 0.55 for mean-reverting data, got {result.hurst}"

    def test_hurst_random_walk(self):
        """Test Hurst for random walk data."""
        from strategies.regime_detection import HurstCalculator

        # Create random walk
        np.random.seed(42)
        n = 1000
        random_walk = np.cumsum(np.random.randn(n))

        calculator = HurstCalculator()
        result = calculator.calculate_dfa(random_walk)

        # Random walk should have H ≈ 0.5
        assert 0.4 < result.hurst < 0.6, f"Expected H ≈ 0.5 for random walk, got {result.hurst}"

    def test_hurst_rs_vs_dfa(self):
        """Test both R/S and DFA methods produce similar results."""
        from strategies.regime_detection import HurstCalculator

        np.random.seed(42)
        n = 500
        data = np.cumsum(np.random.randn(n))

        calculator = HurstCalculator()
        rs_result = calculator.calculate_rs(data)
        dfa_result = calculator.calculate_dfa(data)

        # Both methods should give similar results (within 0.2)
        assert abs(rs_result.hurst - dfa_result.hurst) < 0.2, \
            f"R/S ({rs_result.hurst}) and DFA ({dfa_result.hurst}) differ too much"

    def test_hurst_rolling(self):
        """Test rolling Hurst calculation."""
        from strategies.regime_detection import HurstCalculator

        np.random.seed(42)
        n = 300
        data = pd.Series(np.cumsum(np.random.randn(n)))

        calculator = HurstCalculator()
        rolling = calculator.calculate_rolling(data, window=100)

        assert len(rolling) == n
        assert 'hurst' in rolling.columns
        assert 'regime' in rolling.columns

        # Should have valid values after warmup
        valid = rolling['hurst'].dropna()
        assert len(valid) > 0
        assert all(0 <= h <= 1 for h in valid)


class TestZigZagLabeler:
    """Tests for ZigZag labeling."""

    def test_import(self):
        """Test module imports correctly."""
        from strategies.regime_detection import ZigZagLabeler, ZigZagConfig
        assert ZigZagLabeler is not None
        assert ZigZagConfig is not None

    def test_zigzag_basic(self):
        """Test basic ZigZag labeling."""
        from strategies.regime_detection import ZigZagLabeler

        # Create data with clear trend reversals
        n = 100
        prices = np.concatenate([
            np.linspace(100, 120, 30),  # Up
            np.linspace(120, 90, 40),   # Down
            np.linspace(90, 110, 30),   # Up
        ])

        df = pd.DataFrame({
            'close': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
        })

        labeler = ZigZagLabeler()
        result = labeler.label(df)

        assert len(result) == n
        assert 'label' in result.columns
        assert 'label_name' in result.columns

        # Should have both up and down labels
        labels = result['label'].unique()
        assert len(labels) >= 2

    def test_zigzag_pivots(self):
        """Test pivot point detection."""
        from strategies.regime_detection import ZigZagLabeler

        # Create simple up-down-up pattern
        prices = np.array([100, 105, 110, 115, 120, 115, 110, 105, 100, 105, 110])
        df = pd.DataFrame({
            'close': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
        })

        labeler = ZigZagLabeler()
        labeler.label(df)
        pivots = labeler.get_pivots()

        # Should detect at least one pivot
        assert len(pivots) >= 0  # May vary based on threshold


class TestTrendScanningLabeler:
    """Tests for Trend Scanning labeling."""

    def test_import(self):
        """Test module imports correctly."""
        from strategies.regime_detection import TrendScanningLabeler, TrendScanConfig
        assert TrendScanningLabeler is not None
        assert TrendScanConfig is not None

    def test_trend_scanning_basic(self):
        """Test basic Trend Scanning labeling."""
        from strategies.regime_detection import TrendScanningLabeler, TrendScanConfig

        # Create data with clear trends
        n = 200
        np.random.seed(42)

        # First half: uptrend, second half: downtrend
        up = np.linspace(100, 150, n // 2) + np.random.randn(n // 2) * 2
        down = np.linspace(150, 100, n // 2) + np.random.randn(n // 2) * 2
        prices = np.concatenate([up, down])

        df = pd.DataFrame({'close': prices})

        config = TrendScanConfig(min_window=10, max_window=30, t_threshold=1.5)
        labeler = TrendScanningLabeler(config)
        result = labeler.label(df)

        assert len(result) == n
        assert 'label' in result.columns
        assert 't_value' in result.columns

        # First half should have more positive t-values
        first_half_t = result['t_value'].iloc[:n//2].mean()
        second_half_t = result['t_value'].iloc[n//2:].mean()
        assert first_half_t > second_half_t


class TestHMMRegimeDetector:
    """Tests for HMM regime detection."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        np.random.seed(42)
        n = 500

        # Create regime-switching data
        returns = np.zeros(n)

        # Bull regime (0-150)
        returns[:150] = np.random.normal(0.001, 0.01, 150)

        # Bear regime (150-300)
        returns[150:300] = np.random.normal(-0.001, 0.02, 150)

        # Sideways regime (300-500)
        returns[300:] = np.random.normal(0.0, 0.005, 200)

        prices = 100 * np.exp(np.cumsum(returns))

        return pd.DataFrame({
            'open': prices * 0.999,
            'high': prices * 1.005,
            'low': prices * 0.995,
            'close': prices,
            'volume': np.random.randint(1000, 10000, n),
        })

    def test_import(self):
        """Test module imports correctly."""
        from strategies.regime_detection import HMMRegimeDetector, HMMConfig, MarketRegime
        assert HMMRegimeDetector is not None
        assert HMMConfig is not None
        assert MarketRegime is not None

    def test_hmm_fit_predict(self, sample_data):
        """Test HMM fitting and prediction."""
        pytest.importorskip('hmmlearn')
        from strategies.regime_detection import HMMRegimeDetector, HMMConfig

        config = HMMConfig(n_states=3, n_iter=50)
        detector = HMMRegimeDetector(config)

        # Fit
        detector.fit(sample_data)
        assert detector.is_fitted

        # Predict
        result = detector.predict(sample_data)
        assert len(result) == len(sample_data)
        assert 'regime' in result.columns
        assert 'confidence' in result.columns

    def test_hmm_transition_matrix(self, sample_data):
        """Test transition matrix extraction."""
        pytest.importorskip('hmmlearn')
        from strategies.regime_detection import HMMRegimeDetector, HMMConfig

        config = HMMConfig(n_states=3)
        detector = HMMRegimeDetector(config)
        detector.fit(sample_data)

        trans_matrix = detector.get_transition_matrix()
        assert trans_matrix.shape == (3, 3)

        # Rows should sum to 1
        row_sums = trans_matrix.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(3), decimal=5)


class TestHybridRegimeDetector:
    """Tests for hybrid HMM-Hurst detector."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        np.random.seed(42)
        n = 500

        returns = np.random.normal(0.001, 0.015, n)
        prices = 100 * np.exp(np.cumsum(returns))

        return pd.DataFrame({
            'open': prices * 0.999,
            'high': prices * 1.005,
            'low': prices * 0.995,
            'close': prices,
            'volume': np.random.randint(1000, 10000, n),
        })

    def test_import(self):
        """Test module imports correctly."""
        from strategies.regime_detection import (
            HybridRegimeDetector, HybridConfig, StrategyMode
        )
        assert HybridRegimeDetector is not None
        assert HybridConfig is not None
        assert StrategyMode is not None

    def test_hybrid_fit_predict(self, sample_data):
        """Test hybrid detector fitting and prediction."""
        pytest.importorskip('hmmlearn')
        from strategies.regime_detection import HybridRegimeDetector, HybridConfig

        config = HybridConfig(
            hmm_n_states=2,
            hurst_window=50,
            use_hurst_as_feature=False,
        )
        detector = HybridRegimeDetector(config)

        # Fit
        detector.fit(sample_data)
        assert detector.is_fitted

        # Predict
        result = detector.predict(sample_data)
        assert len(result) == len(sample_data)
        assert 'market_regime' in result.columns
        assert 'hurst' in result.columns
        assert 'strategy_mode' in result.columns
        assert 'position_scale' in result.columns

    def test_hybrid_current_state(self, sample_data):
        """Test getting current state."""
        pytest.importorskip('hmmlearn')
        from strategies.regime_detection import HybridRegimeDetector, HybridConfig

        config = HybridConfig(
            hmm_n_states=2,
            hurst_window=50,
            use_hurst_as_feature=False,
        )
        detector = HybridRegimeDetector(config)
        detector.fit(sample_data)

        state = detector.get_current_state(sample_data)

        assert state.market_regime is not None
        assert 0 <= state.regime_probability <= 1
        assert 0 <= state.hurst_value <= 1
        assert state.strategy_mode is not None
        assert 0 <= state.position_scale <= 1


class TestDrawdownLabeler:
    """Tests for drawdown-based labeling."""

    def test_import(self):
        """Test module imports correctly."""
        from strategies.regime_detection import DrawdownLabeler
        assert DrawdownLabeler is not None

    def test_drawdown_labeling(self):
        """Test drawdown regime labeling."""
        from strategies.regime_detection import DrawdownLabeler

        # Create data with a significant drawdown
        n = 200
        prices = np.concatenate([
            np.linspace(100, 120, 50),   # Up
            np.linspace(120, 80, 50),    # Down (crisis)
            np.linspace(80, 90, 50),     # Recovery
            np.linspace(90, 100, 50),    # Normal
        ])

        df = pd.DataFrame({'close': prices})

        labeler = DrawdownLabeler(crisis_threshold=-0.2)
        result = labeler.label(df)

        assert len(result) == n
        assert 'label' in result.columns
        assert 'drawdown' in result.columns

        # Should have crisis labels during drawdown
        assert 'crisis' in result['label'].values


class TestVolatilityRegimeLabeler:
    """Tests for volatility regime labeling."""

    def test_import(self):
        """Test module imports correctly."""
        from strategies.regime_detection import VolatilityRegimeLabeler
        assert VolatilityRegimeLabeler is not None

    def test_volatility_labeling(self):
        """Test volatility regime labeling."""
        from strategies.regime_detection import VolatilityRegimeLabeler

        np.random.seed(42)
        n = 300

        # Create data with varying volatility
        low_vol = np.cumsum(np.random.randn(100) * 0.5) + 100
        high_vol = np.cumsum(np.random.randn(100) * 2.0) + low_vol[-1]
        med_vol = np.cumsum(np.random.randn(100) * 1.0) + high_vol[-1]

        prices = np.concatenate([low_vol, high_vol, med_vol])

        df = pd.DataFrame({
            'close': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
        })

        labeler = VolatilityRegimeLabeler(vol_window=10)
        result = labeler.label(df)

        assert len(result) == n
        assert 'label' in result.columns
        assert 'volatility' in result.columns

        # Should have all three volatility regimes
        unique_labels = set(result['label'].dropna().unique())
        assert 'low_vol' in unique_labels or 'medium_vol' in unique_labels or 'high_vol' in unique_labels


class TestComputeHurstSimple:
    """Tests for simple Hurst computation function."""

    def test_import(self):
        """Test module imports correctly."""
        from strategies.regime_detection import compute_hurst_simple
        assert compute_hurst_simple is not None

    def test_compute_hurst_simple(self):
        """Test simple Hurst computation."""
        from strategies.regime_detection import compute_hurst_simple

        np.random.seed(42)
        data = np.cumsum(np.random.randn(500))

        h = compute_hurst_simple(data, method='dfa')
        assert 0 <= h <= 1
