import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from src.core.models import Portfolio, SectorAllocation, SectorAnalysisResult
from src.core.data_manager import DataManager
from src.core.session_state import SessionState

class SectorAnalyzer:
    """
    Core engine for analyzing portfolio allocations and performance by sector.
    """
    
    @staticmethod
    def get_sector_classification(tickers: List[str]) -> Dict[str, str]:
        """
        Get sector classification for a list of tickers, using caching to minimize API calls.
        
        Args:
            tickers: List of stock ticker symbols
            
        Returns:
            Dictionary mapping tickers to their sector names.
        """
        sector_mapping = {}
        sector_cache = SessionState.get_sector_cache()
        
        for ticker in tickers:
            if ticker in sector_cache:
                sector_mapping[ticker] = sector_cache[ticker]
            else:
                sector = DataManager.get_sector_info(ticker)
                sector_mapping[ticker] = sector
                SessionState.cache_sector(ticker, sector)
                
        return sector_mapping
        
    @staticmethod
    def calculate_sector_allocation(portfolio: Portfolio, sector_mapping: Dict[str, str]) -> List[SectorAllocation]:
        """
        Calculate the portfolio allocation by sector.
        
        Args:
            portfolio: Portfolio object
            sector_mapping: Dict mapping tickers to sectors
            
        Returns:
            List of SectorAllocation objects
        """
        sector_weights = {}
        sector_counts = {}
        
        for ticker, weight in portfolio.stocks.items():
            sector = sector_mapping.get(ticker, "Unknown")
            
            if sector not in sector_weights:
                sector_weights[sector] = 0.0
                sector_counts[sector] = 0
                
            sector_weights[sector] += weight
            sector_counts[sector] += 1
            
        allocations = []
        for sector, weight in sector_weights.items():
            allocations.append(SectorAllocation(
                sector=sector,
                weight=weight,
                stock_count=sector_counts[sector]
            ))
            
        # Sort by weight descending
        allocations.sort(key=lambda x: x.weight, reverse=True)
        return allocations
        
    @staticmethod
    def calculate_sector_performance(
        portfolio: Portfolio, 
        stock_returns: pd.DataFrame, 
        sector_mapping: Dict[str, str]
    ) -> Dict[str, float]:
        """
        Calculate performance (return) contribution of each sector to the portfolio.
        
        Args:
            portfolio: Portfolio object
            stock_returns: DataFrame of daily stock returns
            sector_mapping: Dict mapping tickers to sectors
            
        Returns:
            Dict mapping sectors to their average return contribution.
        """
        # Calculate total period return for each stock: prod(1+r) - 1
        stock_period_returns = {}
        for ticker in portfolio.stocks.keys():
            if ticker in stock_returns.columns and not stock_returns[ticker].empty:
                stock_period_returns[ticker] = np.prod(1 + stock_returns[ticker]) - 1.0
            else:
                stock_period_returns[ticker] = 0.0
                
        sector_returns = {}
        
        for ticker, weight in portfolio.stocks.items():
            sector = sector_mapping.get(ticker, "Unknown")
            
            if sector not in sector_returns:
                sector_returns[sector] = 0.0
                
            # Weighted return contribution
            sector_returns[sector] += weight * stock_period_returns[ticker]
            
        return sector_returns
        
    @staticmethod
    def calculate_sector_correlation(stock_returns: pd.DataFrame, sector_mapping: Dict[str, str]) -> Optional[pd.DataFrame]:
        """
        Calculate correlation matrix between different sectors.
        
        Args:
            stock_returns: DataFrame of daily stock returns
            sector_mapping: Dict mapping tickers to sectors
            
        Returns:
            DataFrame representing correlation matrix, or None if insufficient data
        """
        if stock_returns.empty:
            return None
            
        # Group daily returns by sector (equal weighting within sector for the index)
        sector_daily_returns = {}
        sectors_found = set(sector_mapping.values())
        
        for sector in sectors_found:
            sector_tickers = [t for t, s in sector_mapping.items() if s == sector and t in stock_returns.columns]
            if sector_tickers:
                sector_daily_returns[sector] = stock_returns[sector_tickers].mean(axis=1)
                
        if not sector_daily_returns:
            return None
            
        sector_df = pd.DataFrame(sector_daily_returns)
        if len(sector_df.columns) < 2:
            return pd.DataFrame([[1.0]], index=sector_df.columns, columns=sector_df.columns)
            
        corr_matrix = sector_df.corr()
        return corr_matrix
        
    @staticmethod
    def analyze_portfolio(portfolio: Portfolio, stock_returns: pd.DataFrame) -> SectorAnalysisResult:
        """
        Perform complete sector analysis on a portfolio.
        
        Args:
            portfolio: Portfolio object
            stock_returns: DataFrame of daily stock returns
            
        Returns:
            SectorAnalysisResult object
        """
        tickers = portfolio.get_tickers()
        sector_mapping = SectorAnalyzer.get_sector_classification(tickers)
        
        allocations = SectorAnalyzer.calculate_sector_allocation(portfolio, sector_mapping)
        sector_returns = SectorAnalyzer.calculate_sector_performance(portfolio, stock_returns, sector_mapping)
        correlation_matrix = SectorAnalyzer.calculate_sector_correlation(stock_returns, sector_mapping)
        
        return SectorAnalysisResult(
            allocations=allocations,
            sector_returns=sector_returns,
            correlation_matrix=correlation_matrix
        )
