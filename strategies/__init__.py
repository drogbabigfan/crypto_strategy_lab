# Strategies package

# LPPLS (Log-Periodic Power Law Singularity) for bubble detection
from .lppls import (
    LPPLSModel,
    LPPLSParams,
    LPPLSResult,
    fit_lppls,
    LPPLSIndicator,
    WindowConfig,
    FilterConfig,
    ConfidenceResult,
    get_bubble_signal,
)
