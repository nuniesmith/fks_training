"""Flat aggregator module for shared utilities.

<<<<<<< HEAD:src/fks_shared_python/__init__.py
Canonical import path: ``shared_python``.
Legacy alias supported (deprecated): ``shared_python``.
=======
Import path stability: external projects can simply do::

    import shared_python as sp
    from shared_python import get_settings, TradeSignal

All underlying logic lives as sibling modules (config.py, types.py, etc.) so this
file just re-exports a curated surface. Avoid service‑specific symbols here.
>>>>>>> eb1dfce (chore: sync submodule):src/shared_python.py
"""
from config import get_settings, reload_settings_cache, initialize_fks_service  # type: ignore
from logging import init_logging, get_logger  # type: ignore
from exceptions import (  # type: ignore
    RiskLimitExceeded,
    DataFetchError,
    DataValidationError,
    ModelError,
)
from types import (  # type: ignore
    TradeSignal,
    BaselineModelParams,
    RiskParams,
    PositionSizingResult,
    MarketBar,
)
from utils import (  # type: ignore
    get_risk_threshold,
    enforce_risk,
    zscore_outliers,
    kmeans_regimes,
    bid_ask_spread,
    price_divergence,
    cyclical_encode,
)
from risk import (  # type: ignore
    kelly_fraction,
    fractional_position,
    volatility_target_position,
    correlation_sizing,
    dynamic_confidence_scale,
    composite_position,
    average_true_range,
    atr_trailing_stop,
    min_variance_weights,
    risk_parity_weights,
    regime_uncertainty_score,
    hedge_position_size,
    composite_with_hedge,
)
from metrics import (  # type: ignore
    Trade,
    win_rate_with_costs,
    cumulative_equity,
    max_drawdown,
    calmar_ratio,
    information_ratio,
    cvar,
    sharpe_ratio,
    downside_deviation,
    sortino_ratio,
)
from simulation import (  # type: ignore
    apply_slippage,
    simulate_gbm,
    monte_carlo_pnl,
    ood_score,
)
from runtime import run_app, register_app, list_apps  # type: ignore

__all__ = [
    # config
    "get_settings",
    "reload_settings_cache",
    "initialize_fks_service",
    # logging
    "init_logging",
    "get_logger",
    # exceptions
    "RiskLimitExceeded",
    "DataFetchError",
    "DataValidationError",
    "ModelError",
    # types
    "TradeSignal",
    "BaselineModelParams",
    "RiskParams",
    "PositionSizingResult",
    "MarketBar",
    # utils
    "get_risk_threshold",
    "enforce_risk",
    "zscore_outliers",
    "kmeans_regimes",
    "bid_ask_spread",
    "price_divergence",
    "cyclical_encode",
    # risk
    "kelly_fraction",
    "fractional_position",
    "volatility_target_position",
    "correlation_sizing",
    "dynamic_confidence_scale",
    "composite_position",
    "average_true_range",
    "atr_trailing_stop",
    "min_variance_weights",
    "risk_parity_weights",
    "regime_uncertainty_score",
    "hedge_position_size",
    "composite_with_hedge",
    # metrics
    "Trade",
    "win_rate_with_costs",
    "cumulative_equity",
    "max_drawdown",
    "calmar_ratio",
    "information_ratio",
    "cvar",
    "sharpe_ratio",
    "downside_deviation",
    "sortino_ratio",
    # simulation
    "apply_slippage",
    "simulate_gbm",
    "monte_carlo_pnl",
    "ood_score",
    # runtime
    "run_app",
    "register_app",
    "list_apps",
]
