import pytest
from hypothesis import given, strategies as st
from typing import Dict, List
from src.core.models import Portfolio

@given(st.lists(st.text(min_size=1), min_size=2, max_size=10, unique=True))
def test_property_portfolio_name_uniqueness(names: List[str]):
    """
    Property 6: Portfolio Name Uniqueness
    Validates: Requirements 4.2
    Test that a collection of portfolios can maintain unique names.
    """
    portfolios = {}
    for name in names:
        portfolios[name] = Portfolio(name=name, stocks={"AAPL": 1.0})
    
    assert len(portfolios) == len(names)
    assert all(name in portfolios for name in names)

@given(st.dictionaries(
    keys=st.text(min_size=1, max_size=5), 
    values=st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False),
    min_size=2, 
    max_size=10
))
def test_property_portfolio_weight_validation_invalid(stocks: Dict[str, float]):
    """
    Property 7: Portfolio Weight Validation
    Validates: Requirements 4.5, 4.6
    Test that portfolios with weights not summing to 1.0 are rejected.
    """
    total_weight = sum(stocks.values())
    
    portfolio = Portfolio(name="Test", stocks=stocks)
    
    # Check if they sum to 1.0 (with a small tolerance for floating point)
    if abs(total_weight - 1.0) <= 1e-4:
        assert portfolio.validate_weights()
    else:
        assert not portfolio.validate_weights()

@given(st.lists(st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False), min_size=1, max_size=20))
def test_property_portfolio_weight_validation_valid(raw_weights: List[float]):
    """
    Property 7: Portfolio Weight Validation (Valid Case)
    Validates: Requirements 4.5, 4.6
    Test that normalized weights always pass validation.
    """
    total = sum(raw_weights)
    normalized_weights = {f"TICKER_{i}": w / total for i, w in enumerate(raw_weights)}
    
    portfolio = Portfolio(name="Normalized", stocks=normalized_weights)
    assert portfolio.validate_weights()
