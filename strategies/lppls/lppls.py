"""
LPPLS (Log-Periodic Power Law Singularity) Model Implementation.

Based on the JLS (Johansen-Ledoit-Sornette) model for financial bubble detection.
Implements the Filimonov-Sornette (2013) procedure for stable parameter estimation.

LPPLS Equation:
    E[ln p(t)] = A + B(tc - t)^m + C(tc - t)^m * cos(omega * ln(tc - t) - phi)

Where:
    - tc: Critical time (expected crash/regime change time)
    - m: Power law exponent (0 < m < 1 for super-exponential growth)
    - omega: Angular frequency of log-periodic oscillations
    - A: Log price at tc
    - B: Amplitude of power law growth (B < 0 for positive bubble)
    - C: Amplitude of oscillations
    - phi: Phase of oscillations

Filimonov-Sornette Procedure:
    Separates parameters into:
    - Nonlinear: tc, m, omega (searched via optimization)
    - Linear: A, B, C1, C2 (solved via OLS given nonlinear params)
    Where C1 = C*cos(phi), C2 = C*sin(phi)
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, differential_evolution


@dataclass
class LPPLSParams:
    """LPPLS model parameters."""
    tc: float      # Critical time
    m: float       # Power law exponent
    omega: float   # Angular frequency
    A: float       # Log price at tc
    B: float       # Power law amplitude
    C: float       # Oscillation amplitude
    phi: float     # Phase

    @property
    def C1(self) -> float:
        """C1 = C * cos(phi)"""
        return self.C * np.cos(self.phi)

    @property
    def C2(self) -> float:
        """C2 = C * sin(phi)"""
        return self.C * np.sin(self.phi)


@dataclass
class LPPLSResult:
    """LPPLS fitting result."""
    params: LPPLSParams
    ssr: float              # Sum of squared residuals
    r_squared: float        # R-squared
    t1: int                 # Start index of fitting window
    t2: int                 # End index of fitting window
    converged: bool         # Whether optimization converged
    damping: float          # Damping condition D = m|B| / (omega|C|)
    n_oscillations: float   # Number of oscillations in window


class LPPLSModel:
    """
    LPPLS model for bubble detection.

    Uses Filimonov-Sornette procedure: 2-stage optimization with
    nonlinear search over (tc, m, omega) and OLS for (A, B, C1, C2).
    """

    # Default parameter bounds (from research document)
    DEFAULT_BOUNDS = {
        'm': (0.1, 0.9),
        'omega': (4.0, 25.0),
    }

    # Stricter bounds for filtering
    STRICT_BOUNDS = {
        'm': (0.3, 0.7),
        'omega': (6.0, 13.0),
    }

    def __init__(
        self,
        tc_min_days: int = 1,
        tc_max_days: int = 60,
        use_strict_bounds: bool = False,
    ):
        """
        Initialize LPPLS model.

        Args:
            tc_min_days: Minimum days ahead for tc prediction
            tc_max_days: Maximum days ahead for tc prediction
            use_strict_bounds: Use stricter parameter bounds
        """
        self.tc_min_days = tc_min_days
        self.tc_max_days = tc_max_days
        self.bounds = self.STRICT_BOUNDS if use_strict_bounds else self.DEFAULT_BOUNDS

    def fit(
        self,
        log_prices: NDArray[np.float64],
        method: str = 'nelder-mead',
        n_starts: int = 10,
    ) -> Optional[LPPLSResult]:
        """
        Fit LPPLS model to log price data.

        Args:
            log_prices: Array of log prices (ln(price))
            method: Optimization method ('nelder-mead', 'differential-evolution')
            n_starts: Number of random starts for optimization

        Returns:
            LPPLSResult if fitting succeeds, None otherwise
        """
        n = len(log_prices)
        if n < 20:
            return None

        t = np.arange(n, dtype=np.float64)
        y = log_prices

        # tc bounds: from current time + tc_min_days to + tc_max_days
        tc_min = float(n + self.tc_min_days)
        tc_max = float(n + self.tc_max_days)

        bounds = [
            (tc_min, tc_max),                    # tc
            self.bounds['m'],                    # m
            self.bounds['omega'],                # omega
        ]

        best_result = None
        best_ssr = np.inf

        if method == 'differential-evolution':
            # Global optimization
            result = differential_evolution(
                lambda x: self._cost_function(x, t, y),
                bounds=bounds,
                maxiter=500,
                tol=1e-8,
                seed=42,
            )
            if result.success or result.fun < np.inf:
                best_params = result.x
                best_ssr = result.fun
                best_result = self._build_result(best_params, t, y, best_ssr)
        else:
            # Multiple random starts with Nelder-Mead
            for _ in range(n_starts):
                # Random initial point within bounds
                x0 = np.array([
                    np.random.uniform(bounds[0][0], bounds[0][1]),
                    np.random.uniform(bounds[1][0], bounds[1][1]),
                    np.random.uniform(bounds[2][0], bounds[2][1]),
                ])

                try:
                    result = minimize(
                        lambda x: self._cost_function(x, t, y),
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8},
                    )

                    if result.fun < best_ssr:
                        # Check if within bounds
                        tc, m, omega = result.x
                        if (bounds[0][0] <= tc <= bounds[0][1] and
                            bounds[1][0] <= m <= bounds[1][1] and
                            bounds[2][0] <= omega <= bounds[2][1]):
                            best_ssr = result.fun
                            best_result = self._build_result(result.x, t, y, best_ssr)
                except Exception:
                    continue

        return best_result

    def _cost_function(
        self,
        params: NDArray[np.float64],
        t: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> float:
        """
        Cost function for optimization (SSR).

        Given nonlinear params (tc, m, omega), solves for linear params
        (A, B, C1, C2) using OLS, then returns SSR.
        """
        tc, m, omega = params

        # Check basic constraints
        if tc <= t[-1] or m <= 0 or m >= 1 or omega <= 0:
            return np.inf

        try:
            # Solve linear parameters via OLS
            linear_params, ssr = self._solve_linear_params(tc, m, omega, t, y)
            if linear_params is None:
                return np.inf
            return ssr
        except Exception:
            return np.inf

    def _solve_linear_params(
        self,
        tc: float,
        m: float,
        omega: float,
        t: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> Tuple[Optional[NDArray[np.float64]], float]:
        """
        Solve linear parameters (A, B, C1, C2) via OLS.

        Transformed equation:
        ln(p(t)) = A + B*f(t) + C1*g(t) + C2*h(t)

        where:
            f(t) = (tc - t)^m
            g(t) = (tc - t)^m * cos(omega * ln(tc - t))
            h(t) = (tc - t)^m * sin(omega * ln(tc - t))
        """
        dt = tc - t

        # Filter out invalid values
        valid = dt > 0
        if not np.all(valid):
            return None, np.inf

        # Compute basis functions
        dt_m = np.power(dt, m)
        log_dt = np.log(dt)
        omega_log_dt = omega * log_dt

        f = dt_m
        g = dt_m * np.cos(omega_log_dt)
        h = dt_m * np.sin(omega_log_dt)

        # Design matrix X = [1, f, g, h]
        X = np.column_stack([np.ones_like(t), f, g, h])

        try:
            # OLS: beta = (X'X)^-1 X'y
            XtX = X.T @ X
            Xty = X.T @ y
            beta = np.linalg.solve(XtX, Xty)

            # SSR
            residuals = y - X @ beta
            ssr = np.sum(residuals ** 2)

            return beta, ssr
        except np.linalg.LinAlgError:
            return None, np.inf

    def _build_result(
        self,
        nonlinear_params: NDArray[np.float64],
        t: NDArray[np.float64],
        y: NDArray[np.float64],
        ssr: float,
    ) -> LPPLSResult:
        """Build LPPLSResult from optimization output."""
        tc, m, omega = nonlinear_params
        linear_params, _ = self._solve_linear_params(tc, m, omega, t, y)

        A, B, C1, C2 = linear_params

        # Recover C and phi from C1, C2
        C = np.sqrt(C1**2 + C2**2)
        phi = np.arctan2(C2, C1)

        # Compute R-squared
        y_mean = np.mean(y)
        ss_tot = np.sum((y - y_mean) ** 2)
        r_squared = 1 - ssr / ss_tot if ss_tot > 0 else 0.0

        # Damping condition: D = m|B| / (omega|C|)
        damping = (m * np.abs(B)) / (omega * np.abs(C)) if C != 0 else np.inf

        # Number of oscillations in window
        t1, t2 = int(t[0]), int(t[-1])
        dt1 = tc - t1
        dt2 = tc - t2
        if dt1 > 0 and dt2 > 0:
            n_oscillations = (omega / (2 * np.pi)) * np.log(dt1 / dt2)
        else:
            n_oscillations = 0.0

        params = LPPLSParams(
            tc=tc,
            m=m,
            omega=omega,
            A=A,
            B=B,
            C=C,
            phi=phi,
        )

        return LPPLSResult(
            params=params,
            ssr=ssr,
            r_squared=r_squared,
            t1=t1,
            t2=t2,
            converged=True,
            damping=damping,
            n_oscillations=n_oscillations,
        )

    def predict(
        self,
        params: LPPLSParams,
        t: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Predict log prices using LPPLS model.

        Args:
            params: LPPLS parameters
            t: Time indices for prediction

        Returns:
            Predicted log prices
        """
        dt = params.tc - t

        # Handle values where t >= tc
        valid = dt > 0
        result = np.full_like(t, params.A, dtype=np.float64)

        if np.any(valid):
            dt_valid = dt[valid]
            dt_m = np.power(dt_valid, params.m)
            omega_log_dt = params.omega * np.log(dt_valid)

            result[valid] = (
                params.A
                + params.B * dt_m
                + params.C * dt_m * np.cos(omega_log_dt - params.phi)
            )

        return result


def fit_lppls(
    prices: NDArray[np.float64],
    use_log: bool = True,
    **kwargs,
) -> Optional[LPPLSResult]:
    """
    Convenience function to fit LPPLS model to price data.

    Args:
        prices: Price array
        use_log: Whether to take log of prices (default True)
        **kwargs: Additional arguments for LPPLSModel

    Returns:
        LPPLSResult if fitting succeeds, None otherwise
    """
    if use_log:
        log_prices = np.log(prices)
    else:
        log_prices = prices

    model = LPPLSModel(**kwargs)
    return model.fit(log_prices)
