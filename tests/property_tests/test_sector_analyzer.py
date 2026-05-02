import pytest
from hypothesis import given, strategies as st
import pandas as pd
import numpy as np
from typing import Dict, List

from src.core.models import Portfolio, SectorAllocation
from src.features.sector_analyzer.analyzer import SectorAnalyzer

# Strategy for creating random portfolios
@st.composite
def portfolio_strategy(draw):
    tickers = draw(st.lists(st.text(min_size=1, max_size=5), min_size=2, max_size=10, unique=True))
    raw_weights = draw(st.lists(st.floats(min_value=0.01, max_value=1.0), min_size=len(tickers), max_size=len(tickers)))
    total = sum(raw_weights)
    weights = {tickers[i]: w/total for i, w in enumerate(raw_weights)}
    return Portfolio("Test", weights)

# Strategy for sector mappings
@st.composite
def sector_mapping_strategy(draw, portfolio):
    tickers = list(portfolio.stocks.keys())
    sectors = ["Technology", "Healthcare", "Financials", "Energy", "Consumer Discretionary"]
    mapping = {}
    for ticker in tickers:
        mapping[ticker] = draw(st.sampled_from(sectors))
    return mapping

@given(portfolio_strategy())
def test_property_sector_allocation_summation(portfolio):
    """
    Property 17: Sector Allocation Summation
    Validates: Requirements 8.1
    Test that sector allocations sum to 100% (or 1.0 in decimal).
    """
    # Create random sector mapping
    sectors = ["Tech", "Health", "Fin", "Energy"]
    mapping = {t: sectors[hash(t) % len(sectors)] for t in portfolio.stocks.keys()}
    
    allocations = SectorAnalyzer.calculate_sector_allocation(portfolio, mapping)
    
    total_weight = sum(alloc.weight for alloc in allocations)
    assert np.isclose(total_weight, 1.0, atol=1e-4)

@given(portfolio_strategy())
def test_property_sector_contribution_summation(portfolio):
    """
    Property 23: Sector Contribution Summation
    Validates: Requirements 9.3
    Test that sector contributions sum to total portfolio return.
    """
    tickers = list(portfolio.stocks.keys())
    
    # Create fake daily returns for tickers
    days = 20
    returns_data = {t: np.random.normal(0.001, 0.02, days) for t in tickers}
    stock_returns = pd.DataFrame(returns_data)
    
    # Create fake sector mapping
    sectors = ["Tech", "Health", "Fin", "Energy"]
    mapping = {t: sectors[hash(t) % len(sectors)] for t in tickers}
    
    # Calculate sector performance
    sector_returns = SectorAnalyzer.calculate_sector_performance(portfolio, stock_returns, mapping)
    
    # Sum of sector returns
    total_sector_return = sum(sector_returns.values())
    
    # Calculate total portfolio return directly
    # Total portfolio return is the weighted sum of individual stock period returns
    stock_period_returns = {t: np.prod(1 + stock_returns[t]) - 1.0 for t in tickers}
    expected_total_return = sum(portfolio.stocks[t] * stock_period_returns[t] for t in tickers)
    
    assert np.isclose(total_sector_return, expected_total_return, atol=1e-4)

@given(st.integers(min_value=3, max_value=10))
def test_property_correlation_matrix_symmetry(num_sectors):
    """
    Property 24: Sector Correlation Matrix Symmetry
    Validates: Requirements 9.8
    Test that correlation matrix is symmetric with diagonal of 1.0.
    """
    days = 30
    tickers = [f"T{i}" for i in range(num_sectors * 2)]
    returns_data = {t: np.random.normal(0, 0.02, days) for t in tickers}
    stock_returns = pd.DataFrame(returns_data)
    
    # Map every 2 tickers to a different sector
    mapping = {tickers[i]: f"S{i//2}" for i in range(len(tickers))}
    
    corr_matrix = SectorAnalyzer.calculate_sector_correlation(stock_returns, mapping)
    
    assert corr_matrix is not None
    # Check shape
    assert corr_matrix.shape == (num_sectors, num_sectors)
    
    # Check symmetry
    assert np.allclose(corr_matrix.values, corr_matrix.values.T, atol=1e-6)
    
    # Check diagonal is 1.0
    assert np.allclose(np.diag(corr_matrix.values), 1.0, atol=1e-6)
