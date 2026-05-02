"""
Probabilistic Analysis Engine
Core calculations: statistics, VaR, joint probabilities, Monte Carlo.
"""
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple


def calculate_full_stats(returns: pd.DataFrame) -> pd.DataFrame:
    """Calculate comprehensive descriptive statistics for stock returns."""
    result = {}
    for ticker in returns.columns:
        r = returns[ticker].dropna()
        sharpe = (r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan
        result[ticker] = {
            "Expected Return (Daily)": r.mean(),
            "Std Dev σ (Daily)": r.std(),
            "Variance": r.var(),
            "Skewness": float(stats.skew(r)),
            "Kurtosis (Excess)": float(stats.kurtosis(r)),
            "Annual Sharpe Ratio": sharpe,
            "Max Daily Gain": r.max(),
            "Max Daily Loss": r.min(),
            "Positive Days %": (r > 0).mean() * 100,
        }
    return pd.DataFrame(result).T


def calculate_all_var(
    returns: pd.DataFrame,
    confidence_levels: List[float] = [0.90, 0.95, 0.99],
) -> pd.DataFrame:
    """Calculate Gaussian and Historical VaR at multiple confidence levels."""
    result = {}
    for ticker in returns.columns:
        r = returns[ticker].dropna()
        mu, sigma = r.mean(), r.std()
        ticker_result = {}
        for conf in confidence_levels:
            label = f"{int(conf * 100)}%"
            z = stats.norm.ppf(1 - conf)
            ticker_result[f"Gaussian VaR {label}"] = mu + z * sigma
            ticker_result[f"Historical VaR {label}"] = float(
                np.percentile(r, (1 - conf) * 100)
            )
        result[ticker] = ticker_result
    return pd.DataFrame(result).T


def calculate_joint_probabilities(
    returns: pd.DataFrame, threshold: float = 0.0
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Calculate individual and pairwise joint probabilities above threshold."""
    tickers = returns.columns.tolist()
    individual = {t: float((returns[t] > threshold).mean()) for t in tickers}
    matrix = pd.DataFrame(index=tickers, columns=tickers, dtype=float)
    for t1 in tickers:
        for t2 in tickers:
            matrix.loc[t1, t2] = float(
                ((returns[t1] > threshold) & (returns[t2] > threshold)).mean()
            )
    return matrix, individual


def run_monte_carlo(
    returns: pd.DataFrame,
    n_simulations: int = 1000,
    horizon: int = 30,
) -> Dict:
    """Run Monte Carlo simulation on an equal-weighted portfolio."""
    port_returns = returns.mean(axis=1).dropna()
    mu = port_returns.mean()
    sigma = port_returns.std()

    rng = np.random.default_rng(42)
    daily_sims = rng.normal(mu, sigma, size=(n_simulations, horizon))
    cumulative = np.cumprod(1 + daily_sims, axis=1) - 1

    final_values = cumulative[:, -1]
    percentiles = {p: np.percentile(cumulative, p, axis=0) for p in [5, 25, 50, 75, 95]}
    var_95 = float(np.percentile(final_values, 5))

    return {
        "sample_paths": cumulative[:200],
        "percentiles": percentiles,
        "final_values": final_values,
        "mu": mu,
        "sigma": sigma,
        "horizon": horizon,
        "n_simulations": n_simulations,
        "prob_profit": float((final_values > 0).mean()),
        "expected_return": float(final_values.mean()),
        "var_95": var_95,
        "cvar_95": float(final_values[final_values <= var_95].mean()),
        "best_case": float(np.percentile(final_values, 95)),
        "worst_case": float(np.percentile(final_values, 5)),
    }
