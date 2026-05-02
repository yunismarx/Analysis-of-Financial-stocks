import pytest
import pandas as pd
import numpy as np
from src.features.portfolio_comparator.comparator import PortfolioComparator
from src.core.models import Portfolio

class TestPortfolioComparator:
    
    def setup_method(self):
        self.comparator = PortfolioComparator(risk_free_rate=0.02)
        self.portfolio1 = Portfolio(name="P1", stocks={"AAPL": 0.6, "MSFT": 0.4})
        self.portfolio2 = Portfolio(name="P2", stocks={"GOOGL": 1.0})
        self.invalid_portfolio = Portfolio(name="Invalid", stocks={"AAPL": 0.5})
        
    def test_add_portfolio(self):
        assert self.comparator.add_portfolio(self.portfolio1) is True
        assert len(self.comparator.portfolios) == 1
        
        # Duplicate name
        duplicate = Portfolio(name="P1", stocks={"GOOGL": 1.0})
        assert self.comparator.add_portfolio(duplicate) is False
        
        # Invalid weights
        assert self.comparator.add_portfolio(self.invalid_portfolio) is False
        
    def test_remove_portfolio(self):
        self.comparator.add_portfolio(self.portfolio1)
        assert self.comparator.remove_portfolio("P1") is True
        assert len(self.comparator.portfolios) == 0
        assert self.comparator.remove_portfolio("NonExistent") is False
        
    def test_calculate_portfolio_returns(self):
        stock_returns = pd.DataFrame({
            "AAPL": [0.01, -0.01, 0.02],
            "MSFT": [0.02, 0.00, -0.01],
            "GOOGL": [0.00, 0.01, 0.01]
        })
        
        ret = self.comparator.calculate_portfolio_returns(self.portfolio1, stock_returns)
        
        # Day 1: 0.6*0.01 + 0.4*0.02 = 0.006 + 0.008 = 0.014
        assert np.isclose(ret.iloc[0], 0.014)
        
        # Day 2: 0.6*-0.01 + 0.4*0.00 = -0.006
        assert np.isclose(ret.iloc[1], -0.006)
        
    def test_calculate_portfolio_returns_missing_stock(self):
        # MSFT is missing, so AAPL gets 100% weight for P1
        stock_returns = pd.DataFrame({
            "AAPL": [0.01, -0.01, 0.02]
        })
        
        ret = self.comparator.calculate_portfolio_returns(self.portfolio1, stock_returns)
        assert np.isclose(ret.iloc[0], 0.01)
        
    def test_metrics_calculation(self):
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02, 0.01, -0.01])
        market_returns = pd.Series([0.005, -0.01, 0.02, -0.005, 0.01, 0.005, -0.005])
        
        metrics = self.comparator.calculate_metrics(returns, market_returns)
        
        assert "Total Return" in metrics
        assert "Annualized Return" in metrics
        assert "Volatility" in metrics
        assert "Sharpe Ratio" in metrics
        assert "VaR (95%)" in metrics
        assert "CVaR (95%)" in metrics
        assert "Max Drawdown" in metrics
        assert "Beta" in metrics
        
    def test_zero_volatility_sharpe(self):
        returns = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0])
        sharpe = self.comparator.calculate_sharpe_ratio(returns)
        assert sharpe == 0.0
        
    def test_empty_series_handling(self):
        empty_series = pd.Series(dtype=float)
        
        assert self.comparator.calculate_total_return(empty_series) == 0.0
        assert self.comparator.calculate_annualized_return(empty_series) == 0.0
        assert self.comparator.calculate_volatility(empty_series) == 0.0
        assert self.comparator.calculate_var(empty_series) == 0.0
        assert self.comparator.calculate_cvar(empty_series) == 0.0
        assert self.comparator.calculate_max_drawdown(empty_series) == 0.0
        assert self.comparator.calculate_beta(empty_series, pd.Series([0.01, 0.02])) == 1.0
