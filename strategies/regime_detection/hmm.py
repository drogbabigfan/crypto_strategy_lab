"""
Hidden Markov Model (HMM) based Regime Detection.

Uses Gaussian HMM to identify hidden market states:
- Bull (Uptrend): High returns, low volatility
- Bear (Downtrend): Negative returns, high volatility
- Sideways: Low returns, varying volatility

Key algorithms:
- Baum-Welch: Learn model parameters (A, B, pi) from data
- Viterbi: Find most likely state sequence (posterior labeling)
- Forward: Real-time filtering for current state probability
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings

try:
    from hmmlearn.hmm import GaussianHMM
    HAS_HMMLEARN = True
except ImportError:
    HAS_HMMLEARN = False
    warnings.warn("hmmlearn not installed. Install with: pip install hmmlearn")


class MarketRegime(Enum):
    """Market regime states."""
    BULL = "bull"           # Uptrend: positive returns, lower vol
    BEAR = "bear"           # Downtrend: negative returns, higher vol
    SIDEWAYS = "sideways"   # Range-bound: near-zero returns


@dataclass
class HMMConfig:
    """Configuration for HMM regime detector."""
    n_states: int = 3                    # Number of hidden states
    n_iter: int = 100                    # Max iterations for Baum-Welch
    covariance_type: str = 'full'        # 'spherical', 'diag', 'full', 'tied'
    random_state: int = 42               # For reproducibility
    min_covar: float = 1e-3              # Minimum covariance (prevents singularity)
    features: List[str] = field(default_factory=lambda: ['returns', 'volatility'])
    vol_window: int = 20                 # Window for volatility calculation
    prob_threshold: float = 0.6          # Min probability to confirm regime


@dataclass
class RegimeState:
    """Current regime state with probabilities."""
    regime: MarketRegime
    probabilities: Dict[MarketRegime, float]
    confidence: float
    timestamp: Optional[pd.Timestamp] = None

    @property
    def is_bull(self) -> bool:
        return self.regime == MarketRegime.BULL

    @property
    def is_bear(self) -> bool:
        return self.regime == MarketRegime.BEAR


class HMMRegimeDetector:
    """
    Hidden Markov Model for market regime detection.

    Uses Gaussian emission probabilities with returns and volatility
    as observation features to identify latent market states.
    """

    def __init__(self, config: Optional[HMMConfig] = None):
        """
        Args:
            config: HMM configuration parameters
        """
        if not HAS_HMMLEARN:
            raise ImportError(
                "hmmlearn is required for HMM regime detection. "
                "Install with: pip install hmmlearn"
            )

        self.config = config or HMMConfig()
        self.model: Optional[GaussianHMM] = None
        self.is_fitted = False
        self._state_mapping: Dict[int, MarketRegime] = {}
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None

    def fit(
        self,
        data: pd.DataFrame,
        features: Optional[List[str]] = None
    ) -> 'HMMRegimeDetector':
        """
        Fit the HMM model to historical data.

        Args:
            data: DataFrame with OHLCV columns
            features: Feature columns to use (default: computed from config)

        Returns:
            self for method chaining
        """
        # Prepare features
        X, valid_idx = self._prepare_features(data, features)

        if len(X) < self.config.n_states * 10:
            raise ValueError(
                f"Insufficient data for {self.config.n_states} states. "
                f"Need at least {self.config.n_states * 10} samples."
            )

        # Normalize features
        self._feature_means = np.mean(X, axis=0)
        self._feature_stds = np.std(X, axis=0) + 1e-10
        X_normalized = (X - self._feature_means) / self._feature_stds

        # Initialize and fit HMM
        self.model = GaussianHMM(
            n_components=self.config.n_states,
            covariance_type=self.config.covariance_type,
            n_iter=self.config.n_iter,
            random_state=self.config.random_state,
            min_covar=self.config.min_covar,
        )

        self.model.fit(X_normalized)
        self.is_fitted = True

        # Map states to regimes based on emission means
        self._map_states_to_regimes()

        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict regime states for given data using Viterbi algorithm.

        This performs posterior (offline) regime detection using the
        entire sequence to find the most likely state path.

        Args:
            data: DataFrame with OHLCV columns

        Returns:
            DataFrame with regime predictions and probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X, valid_idx = self._prepare_features(data)
        X_normalized = (X - self._feature_means) / self._feature_stds

        # Viterbi decoding
        states = self.model.predict(X_normalized)

        # Get state probabilities
        probs = self.model.predict_proba(X_normalized)

        # Build result DataFrame
        n = len(data)
        results = pd.DataFrame(
            index=data.index,
            columns=['regime', 'regime_id', 'prob_bull', 'prob_bear', 'prob_sideways', 'confidence']
        )

        # Fill valid indices
        for i, idx in enumerate(valid_idx):
            state_id = states[i]
            regime = self._state_mapping[state_id]

            results.loc[data.index[idx], 'regime'] = regime.value
            results.loc[data.index[idx], 'regime_id'] = state_id

            for state_idx, reg in self._state_mapping.items():
                col = f'prob_{reg.value}'
                results.loc[data.index[idx], col] = probs[i, state_idx]

            results.loc[data.index[idx], 'confidence'] = probs[i, state_id]

        return results

    def filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Online filtering: predict current state using only past data.

        Uses Forward algorithm to compute P(state_t | obs_1:t).
        This is appropriate for live trading as it doesn't use future info.

        Args:
            data: DataFrame with OHLCV columns

        Returns:
            DataFrame with filtered regime probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before filtering")

        X, valid_idx = self._prepare_features(data)
        X_normalized = (X - self._feature_means) / self._feature_stds

        n = len(data)
        results = pd.DataFrame(
            index=data.index,
            columns=['regime', 'regime_id', 'prob_bull', 'prob_bear', 'prob_sideways', 'confidence']
        )

        # Forward algorithm for sequential filtering
        for i, idx in enumerate(valid_idx):
            # Use only data up to current point
            X_partial = X_normalized[:i + 1]

            # Get filtered probability for last observation
            log_prob, posteriors = self.model.score_samples(X_partial)
            current_probs = posteriors[-1]

            state_id = np.argmax(current_probs)
            regime = self._state_mapping[state_id]

            results.loc[data.index[idx], 'regime'] = regime.value
            results.loc[data.index[idx], 'regime_id'] = state_id

            for state_idx, reg in self._state_mapping.items():
                col = f'prob_{reg.value}'
                results.loc[data.index[idx], col] = current_probs[state_idx]

            results.loc[data.index[idx], 'confidence'] = current_probs[state_id]

        return results

    def get_current_regime(self, data: pd.DataFrame) -> RegimeState:
        """
        Get current regime state (most recent observation).

        Args:
            data: DataFrame with OHLCV columns

        Returns:
            RegimeState with current regime and probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X, valid_idx = self._prepare_features(data)

        if len(X) == 0:
            return RegimeState(
                regime=MarketRegime.SIDEWAYS,
                probabilities={r: 1/3 for r in MarketRegime},
                confidence=0.0
            )

        X_normalized = (X - self._feature_means) / self._feature_stds

        # Get probability for last observation
        _, posteriors = self.model.score_samples(X_normalized)
        current_probs = posteriors[-1]

        state_id = np.argmax(current_probs)
        regime = self._state_mapping[state_id]

        probs = {
            self._state_mapping[i]: current_probs[i]
            for i in range(self.config.n_states)
        }

        return RegimeState(
            regime=regime,
            probabilities=probs,
            confidence=current_probs[state_id],
            timestamp=data.index[-1] if hasattr(data.index, '__getitem__') else None
        )

    def get_transition_matrix(self) -> pd.DataFrame:
        """
        Get the learned state transition probability matrix.

        Returns:
            DataFrame with transition probabilities A[i,j] = P(state_j | state_i)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        regimes = [self._state_mapping[i].value for i in range(self.config.n_states)]

        return pd.DataFrame(
            self.model.transmat_,
            index=regimes,
            columns=regimes
        )

    def get_stationary_distribution(self) -> Dict[MarketRegime, float]:
        """
        Get the stationary (equilibrium) distribution of regimes.

        This represents the long-run proportion of time spent in each regime.

        Returns:
            Dict mapping regime to probability
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        # Compute stationary distribution from transition matrix
        trans = self.model.transmat_
        eigenvalues, eigenvectors = np.linalg.eig(trans.T)

        # Find eigenvector for eigenvalue = 1
        idx = np.argmin(np.abs(eigenvalues - 1))
        stationary = np.real(eigenvectors[:, idx])
        stationary = stationary / stationary.sum()

        return {
            self._state_mapping[i]: stationary[i]
            for i in range(self.config.n_states)
        }

    def _prepare_features(
        self,
        data: pd.DataFrame,
        features: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Prepare feature matrix from OHLCV data.

        Returns:
            Tuple of (feature_matrix, valid_indices)
        """
        df = data.copy()

        # Compute returns if not present
        if 'returns' not in df.columns:
            df['returns'] = df['close'].pct_change()

        # Compute volatility if not present
        if 'volatility' not in df.columns:
            df['volatility'] = df['returns'].rolling(
                window=self.config.vol_window
            ).std()

        # Select features
        feature_cols = features or self.config.features
        X = df[feature_cols].values

        # Find valid (non-NaN) rows
        valid_mask = ~np.any(np.isnan(X), axis=1)
        valid_idx = np.where(valid_mask)[0].tolist()

        return X[valid_mask], valid_idx

    def _map_states_to_regimes(self):
        """
        Map HMM state indices to market regimes based on emission means.

        Bull: highest mean return
        Bear: lowest mean return
        Sideways: middle mean return
        """
        if self.model is None:
            return

        # Get mean returns for each state (assume returns is first feature)
        means = self.model.means_[:, 0]

        # Sort states by mean return
        sorted_states = np.argsort(means)

        # Map: lowest -> Bear, highest -> Bull, middle -> Sideways
        if self.config.n_states == 2:
            self._state_mapping = {
                sorted_states[0]: MarketRegime.BEAR,
                sorted_states[1]: MarketRegime.BULL,
            }
        elif self.config.n_states == 3:
            self._state_mapping = {
                sorted_states[0]: MarketRegime.BEAR,
                sorted_states[1]: MarketRegime.SIDEWAYS,
                sorted_states[2]: MarketRegime.BULL,
            }
        else:
            # For more states, use simple mapping
            for i in range(self.config.n_states):
                if i < self.config.n_states // 3:
                    self._state_mapping[sorted_states[i]] = MarketRegime.BEAR
                elif i >= 2 * self.config.n_states // 3:
                    self._state_mapping[sorted_states[i]] = MarketRegime.BULL
                else:
                    self._state_mapping[sorted_states[i]] = MarketRegime.SIDEWAYS


class GaussianHMMFromScratch:
    """
    Simple Gaussian HMM implementation without hmmlearn dependency.

    This is a fallback implementation for environments where hmmlearn
    is not available. Uses EM algorithm for training.
    """

    def __init__(
        self,
        n_states: int = 3,
        n_iter: int = 100,
        tol: float = 1e-4,
        random_state: int = 42
    ):
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state

        # Model parameters
        self.pi: Optional[np.ndarray] = None      # Initial state distribution
        self.A: Optional[np.ndarray] = None       # Transition matrix
        self.means: Optional[np.ndarray] = None   # Emission means
        self.covars: Optional[np.ndarray] = None  # Emission covariances

    def fit(self, X: np.ndarray) -> 'GaussianHMMFromScratch':
        """
        Fit model using Baum-Welch (EM) algorithm.

        Args:
            X: Observation sequence, shape (T, n_features)

        Returns:
            self for method chaining
        """
        np.random.seed(self.random_state)
        T, n_features = X.shape

        # Initialize parameters
        self._initialize_params(X)

        prev_log_likelihood = -np.inf

        for iteration in range(self.n_iter):
            # E-step: Forward-Backward algorithm
            alpha, beta, gamma, xi, log_likelihood = self._e_step(X)

            # Check convergence
            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                break
            prev_log_likelihood = log_likelihood

            # M-step: Update parameters
            self._m_step(X, gamma, xi)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict most likely state sequence using Viterbi algorithm.

        Args:
            X: Observation sequence, shape (T, n_features)

        Returns:
            Array of state indices
        """
        T = len(X)

        # Compute emission probabilities
        B = self._compute_emission_probs(X)

        # Viterbi algorithm
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)

        # Initialization
        delta[0] = np.log(self.pi + 1e-10) + np.log(B[0] + 1e-10)

        # Recursion
        for t in range(1, T):
            for j in range(self.n_states):
                trans_probs = delta[t - 1] + np.log(self.A[:, j] + 1e-10)
                psi[t, j] = np.argmax(trans_probs)
                delta[t, j] = trans_probs[psi[t, j]] + np.log(B[t, j] + 1e-10)

        # Backtracking
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(delta[-1])

        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute state probabilities using Forward-Backward algorithm.

        Args:
            X: Observation sequence

        Returns:
            Array of shape (T, n_states) with P(state | observations)
        """
        alpha, beta, gamma, _, _ = self._e_step(X)
        return gamma

    def _initialize_params(self, X: np.ndarray):
        """Initialize model parameters using k-means-like initialization."""
        T, n_features = X.shape

        # Initial state distribution (uniform)
        self.pi = np.ones(self.n_states) / self.n_states

        # Transition matrix (slight preference for staying in same state)
        self.A = np.ones((self.n_states, self.n_states)) * 0.1
        np.fill_diagonal(self.A, 0.8)
        self.A /= self.A.sum(axis=1, keepdims=True)

        # Initialize means using data quantiles
        percentiles = np.linspace(0, 100, self.n_states + 2)[1:-1]
        self.means = np.percentile(X, percentiles, axis=0)

        # Initialize covariances
        var = np.var(X, axis=0)
        self.covars = np.array([np.diag(var) for _ in range(self.n_states)])

    def _compute_emission_probs(self, X: np.ndarray) -> np.ndarray:
        """Compute Gaussian emission probabilities B[t,j] = P(x_t | state_j)."""
        T, n_features = X.shape
        B = np.zeros((T, self.n_states))

        for j in range(self.n_states):
            diff = X - self.means[j]
            cov_inv = np.linalg.inv(self.covars[j] + np.eye(n_features) * 1e-6)
            cov_det = np.linalg.det(self.covars[j] + np.eye(n_features) * 1e-6)

            exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
            normalizer = 1 / np.sqrt((2 * np.pi) ** n_features * cov_det)

            B[:, j] = normalizer * np.exp(exponent)

        return B

    def _e_step(self, X: np.ndarray) -> Tuple[np.ndarray, ...]:
        """E-step: Forward-Backward algorithm."""
        T = len(X)
        B = self._compute_emission_probs(X)

        # Forward pass
        alpha = np.zeros((T, self.n_states))
        alpha[0] = self.pi * B[0]
        alpha[0] /= alpha[0].sum() + 1e-10

        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.A) * B[t]
            alpha[t] /= alpha[t].sum() + 1e-10

        # Backward pass
        beta = np.zeros((T, self.n_states))
        beta[-1] = 1

        for t in range(T - 2, -1, -1):
            beta[t] = (self.A * B[t + 1]) @ beta[t + 1]
            beta[t] /= beta[t].sum() + 1e-10

        # Compute gamma and xi
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True) + 1e-10

        xi = np.zeros((T - 1, self.n_states, self.n_states))
        for t in range(T - 1):
            xi[t] = np.outer(alpha[t], beta[t + 1] * B[t + 1]) * self.A
            xi[t] /= xi[t].sum() + 1e-10

        # Log likelihood
        log_likelihood = np.sum(np.log(alpha.sum(axis=1) + 1e-10))

        return alpha, beta, gamma, xi, log_likelihood

    def _m_step(self, X: np.ndarray, gamma: np.ndarray, xi: np.ndarray):
        """M-step: Update parameters."""
        T, n_features = X.shape

        # Update initial distribution
        self.pi = gamma[0]

        # Update transition matrix
        self.A = xi.sum(axis=0)
        self.A /= self.A.sum(axis=1, keepdims=True) + 1e-10

        # Update emission parameters
        for j in range(self.n_states):
            gamma_j = gamma[:, j:j + 1]
            gamma_sum = gamma_j.sum() + 1e-10

            # Update mean
            self.means[j] = (gamma_j * X).sum(axis=0) / gamma_sum

            # Update covariance
            diff = X - self.means[j]
            self.covars[j] = (gamma_j * diff).T @ diff / gamma_sum
            # Add small diagonal for numerical stability
            self.covars[j] += np.eye(n_features) * 1e-3
