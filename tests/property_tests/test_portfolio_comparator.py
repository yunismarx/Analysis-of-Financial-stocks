import pytest
from hypothesis import given, strategies as st
from hypothesis.extra.pandas import series, data_frames, column
import pandas as pd
import numpy as np
from src.features.portfolio_comparator.comparator import PortfolioComparator
from src.core.models import Portfolio

# Generate a series of daily returns bounded to realistic limits
return_strategy = st.floats(min_value=-0.2, max_value=0.2, allow_nan=False, allow_infinity=False)

@given(st.lists(return_strategy, min_size=5, max_size=100))
def test_property_total_return(returns_list):
    """
    Property 8: Total Return Calculation
    Validates: Requirements 5.1
    """
    returns = pd.Series(returns_list)
    comparator = PortfolioComparator()
    
    total_ret = comparator.calculate_total_return(returns)
    expected = np.prod(1 + returns) - 1.0
    
    assert np.isclose(total_ret, expected, atol=1e-6)

@given(st.lists(return_strategy, min_size=10, max_size=100), st.integers(min_value=10, max_value=252))
def test_property_annualized_return(returns_list, periods_per_year):
    """
    Property 9: Annualized Return Calculation
    Validates: Requirements 5.2
    """
    returns = pd.Series(returns_list)
    comparator = PortfolioComparator()
    
    ann_ret = comparator.calculate_annualized_return(returns, periods_per_year=periods_per_year)
    total_return = np.prod(1 + returns) - 1.0
    years = len(returns) / periods_per_year
    expected = (1 + total_return) ** (1 / years) - 1.0
    
    assert np.isclose(ann_ret, expected, atol=1e-6)

@given(st.lists(return_strategy, min_size=5, max_size=100), st.integers(min_value=10, max_value=252))
def test_property_volatility(returns_list, periods_per_year):
    """
    Property 10: Portfolio Volatility Calculation
    Validates: Requirements 5.3
    """
    returns = pd.Series(returns_list)
    comparator = PortfolioComparator()
    
    vol = comparator.calculate_volatility(returns, periods_per_year=periods_per_year)
    expected = returns.std() * np.sqrt(periods_per_year)
    
    assert np.isclose(vol, expected, atol=1e-6)

@given(st.lists(return_strategy, min_size=10, max_size=100), st.floats(min_value=0.0, max_value=0.1))
def test_property_sharpe_ratio(returns_list, risk_free_rate):
    """
    Property 11: Sharpe Ratio Calculation
    Validates: Requirements 5.4
    """
    returns = pd.Series(returns_list)
    comparator = PortfolioComparator(risk_free_rate=risk_free_rate)
    
    sharpe = comparator.calculate_sharpe_ratio(returns)
    
    vol = returns.std() * np.sqrt(252)
    if vol == 0:
        assert sharpe == 0.0
    else:
        ann_ret = comparator.calculate_annualized_return(returns)
        expected = (ann_ret - risk_free_rate) / vol
        assert np.isclose(sharpe, expected, atol=1e-6)

@given(st.lists(return_strategy, min_size=20, max_size=100))
def test_property_var_cvar(returns_list):
    """
    Property 13 & 14: VaR and CVaR Calculation
    Validates: Requirements 6.1, 6.2
    """
    returns = pd.Series(returns_list)
    comparator = PortfolioComparator()
    
    var_95 = comparator.calculate_var(returns, confidence_level=0.95)
    cvar_95 = comparator.calculate_cvar(returns, confidence_level=0.95)
    
    expected_var = np.percentile(returns, 5)
    assert np.isclose(var_95, expected_var, atol=1e-6)
    
    tail = returns[returns <= expected_var]
    if len(tail) > 0:
        expected_cvar = tail.mean()
        assert np.isclose(cvar_95, expected_cvar, atol=1e-6)
    else:
        assert np.isclose(cvar_95, expected_var, atol=1e-6)

@given(st.lists(return_strategy, min_size=10, max_size=100))
def test_property_max_drawdown(returns_list):
    """
    Property 15: Maximum Drawdown Calculation
    Validates: Requirements 6.3
    """
    returns = pd.Series(returns_list)
    comparator = PortfolioComparator()
    
    max_dd = comparator.calculate_max_drawdown(returns)
    
    cum = (1 + returns).cumprod()
    peaks = cum.cummax()
    dds = (cum - peaks) / peaks
    expected = dds.min()
    
    assert np.isclose(max_dd, expected, atol=1e-6)

@given(
    st.lists(return_strategy, min_size=10, max_size=100),
    st.lists(return_strategy, min_size=10, max_size=100)
)
def test_property_beta(port_ret, mkt_ret):
    """
    Property 16: Beta Calculation
    Validates: Requirements 6.4
    """
    # ensure same length
    min_len = min(len(port_ret), len(mkt_ret))
    p = pd.Series(port_ret[:min_len])
    m = pd.Series(mkt_ret[:min_len])
    
    comparator = PortfolioComparator()
    beta = comparator.calculate_beta(p, m)
    
    market_var = m.var()
    if market_var == 0:
        assert beta == 1.0
    else:
        cov = p.cov(m)
        expected = cov / market_var
        assert np.isclose(beta, expected, atol=1e-6)

@given(
    st.dictionaries(
        keys=st.text(min_size=1),
        values=st.lists(return_strategy, min_size=20, max_size=100),
        min_size=2,
        max_size=5
    )
)
def test_property_best_portfolio_selection(returns_dict_raw):
    """
    Property 12: Best Portfolio Selection
    Validates: Requirements 5.7
    """
    # Convert lists to pd.Series and ensure all series have same length (take min length)
    min_len = min(len(r) for r in returns_dict_raw.values())
    returns_dict = {k: pd.Series(v[:min_len]) for k, v in returns_dict_raw.items()}
    
    comparator = PortfolioComparator()
    best_name = comparator.get_best_portfolio(returns_dict)
    
    # Calculate sharpe manually to verify
    sharpes = {}
    for k, v in returns_dict.items():
        if v.empty:
            continue
        vol = v.std() * np.sqrt(252)
        if vol == 0:
            sharpes[k] = 0.0
        else:
            total_ret = np.prod(1 + v) - 1.0
            years = len(v) / 252
            if years == 0:
                ann_ret = 0.0
            else:
                ann_ret = (1 + total_ret) ** (1 / years) - 1.0
            sharpes[k] = (ann_ret - 0.02) / vol
            
    if sharpes:
        expected_best = max(sharpes.keys(), key=lambda k: sharpes[k])
        assert best_name == expected_best
    else:
        assert best_name is None
