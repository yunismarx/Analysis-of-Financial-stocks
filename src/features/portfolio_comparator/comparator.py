import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from src.core.models import Portfolio

class PortfolioComparator:
    """
    Core engine for comparing multiple portfolios and calculating performance metrics.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.portfolios: Dict[str, Portfolio] = {}
        self.risk_free_rate = risk_free_rate
        
    def add_portfolio(self, portfolio: Portfolio) -> bool:
        """
        Add a portfolio to the comparator.
        Rejects if name already exists or if weights are invalid.
        """
        if portfolio.name in self.portfolios:
            return False
            
        if not portfolio.validate_weights():
            return False
            
        self.portfolios[portfolio.name] = portfolio
        return True
        
    def remove_portfolio(self, name: str) -> bool:
        """Remove a portfolio by name."""
        if name in self.portfolios:
            del self.portfolios[name]
            return True
        return False
        
    def calculate_portfolio_returns(self, portfolio: Portfolio, stock_returns: pd.DataFrame) -> pd.Series:
        """
        Calculate weighted daily returns for a portfolio.
        
        Args:
            portfolio: The portfolio to calculate returns for.
            stock_returns: DataFrame containing daily returns for individual stocks.
            
        Returns:
            Series of daily portfolio returns.
        """
        tickers = portfolio.get_tickers()
        
        # Filter available stocks
        available_tickers = [t for t in tickers if t in stock_returns.columns]
        if not available_tickers:
            return pd.Series(dtype=float)
            
        # Re-normalize weights for available stocks
        weights_sum = sum(portfolio.stocks[t] for t in available_tickers)
        if weights_sum == 0:
            return pd.Series(0.0, index=stock_returns.index)
            
        normalized_weights = np.array([portfolio.stocks[t] / weights_sum for t in available_tickers])
        
        # Calculate weighted returns
        returns_subset = stock_returns[available_tickers]
        portfolio_returns = returns_subset.dot(normalized_weights)
        
        return portfolio_returns
        
    def calculate_total_return(self, returns: pd.Series) -> float:
        """Calculate cumulative total return over the period."""
        if returns.empty:
            return 0.0
        # Formula: (1+r1)*(1+r2)*... - 1
        return np.prod(1 + returns) - 1.0
        
    def calculate_annualized_return(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """Calculate annualized return."""
        if returns.empty or len(returns) == 0:
            return 0.0
        total_return = self.calculate_total_return(returns)
        years = len(returns) / periods_per_year
        if years == 0:
            return 0.0
        return (1 + total_return) ** (1 / years) - 1.0
        
    def calculate_volatility(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """Calculate annualized volatility."""
        if returns.empty:
            return 0.0
        return returns.std() * np.sqrt(periods_per_year)
        
    def calculate_sharpe_ratio(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """Calculate annualized Sharpe ratio."""
        volatility = self.calculate_volatility(returns, periods_per_year)
        if volatility == 0.0 or np.isnan(volatility):
            return 0.0
            
        annual_return = self.calculate_annualized_return(returns, periods_per_year)
        return (annual_return - self.risk_free_rate) / volatility
        
    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk (Historical)."""
        if returns.empty:
            return 0.0
        # We calculate the percentile for the left tail. 95% confidence means 5th percentile.
        percentile = (1.0 - confidence_level) * 100
        return float(np.percentile(returns, percentile))
        
    def calculate_cvar(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        if returns.empty:
            return 0.0
        var = self.calculate_var(returns, confidence_level)
        tail_returns = returns[returns <= var]
        if len(tail_returns) == 0:
            return var
        return float(tail_returns.mean())
        
    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if returns.empty:
            return 0.0
            
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdowns = (cumulative - rolling_max) / rolling_max
        return float(drawdowns.min())
        
    def calculate_beta(self, portfolio_returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate beta relative to market index."""
        if portfolio_returns.empty or market_returns.empty:
            return 1.0
            
        # Ensure alignment
        aligned_data = pd.concat([portfolio_returns, market_returns], axis=1).dropna()
        if len(aligned_data) < 2:
            return 1.0
            
        p_ret = aligned_data.iloc[:, 0]
        m_ret = aligned_data.iloc[:, 1]
        
        market_var = m_ret.var()
        if market_var == 0:
            return 1.0
            
        covariance = p_ret.cov(m_ret)
        return float(covariance / market_var)
        
    def calculate_metrics(self, returns: pd.Series, market_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """Calculate all performance metrics for a portfolio's returns."""
        metrics = {
            "Total Return": self.calculate_total_return(returns),
            "Annualized Return": self.calculate_annualized_return(returns),
            "Volatility": self.calculate_volatility(returns),
            "Sharpe Ratio": self.calculate_sharpe_ratio(returns),
            "VaR (95%)": self.calculate_var(returns),
            "CVaR (95%)": self.calculate_cvar(returns),
            "Max Drawdown": self.calculate_max_drawdown(returns),
        }
        
        if market_returns is not None:
            metrics["Beta"] = self.calculate_beta(returns, market_returns)
            
        return metrics

    def get_best_portfolio(self, returns_dict: Dict[str, pd.Series]) -> Optional[str]:
        """
        Identify the best performing portfolio based on Sharpe ratio.
        
        Args:
            returns_dict: Dictionary mapping portfolio names to their daily returns series.
            
        Returns:
            Name of the best portfolio, or None if no valid portfolios provided.
        """
        best_name = None
        best_sharpe = float('-inf')
        
        for name, returns in returns_dict.items():
            if returns.empty:
                continue
            sharpe = self.calculate_sharpe_ratio(returns)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_name = name
                
        return best_name
