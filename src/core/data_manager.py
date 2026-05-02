"""
Data Manager - Centralized data fetching, caching, and validation
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict, Optional


class DataManager:
    """
    Centralized data management for stock data fetching and processing.
    Handles data acquisition, caching, and validation.
    """
    
    def __init__(self):
        """Initialize the DataManager"""
        pass
    
    @staticmethod
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def get_stock_data(
        tickers: List[str], 
        start_date: datetime, 
        end_date: datetime
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Fetch historical price data for given tickers with error handling.
        
        Args:
            tickers: List of stock ticker symbols
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            Tuple of (DataFrame with price data, List of failed tickers)
        """
        successful_data = {}
        failed_tickers = []
        
        for ticker in tickers:
            try:
                data = yf.download(
                    ticker, 
                    start=start_date, 
                    end=end_date, 
                    progress=False
                )
                
                if data.empty:
                    failed_tickers.append(ticker)
                    st.warning(f"⚠️ لا توجد بيانات متاحة للسهم {ticker}")
                else:
                    # Handle missing data: forward-fill up to 5 days, then drop
                    data = data.ffill(limit=5)
                    data = data.dropna()
                    successful_data[ticker] = data
                    
            except Exception as e:
                st.error(f"❌ Failed to load data for {ticker}: {str(e)}")
                failed_tickers.append(ticker)
        
        if not successful_data:
            st.error("❌ Failed to load any stocks. Please check ticker symbols and your internet connection.")
            return pd.DataFrame(), failed_tickers
        
        # Combine data from all successful tickers
        if len(successful_data) == 1:
            # Single ticker - return as DataFrame with ticker as column name
            ticker = list(successful_data.keys())[0]
            df = successful_data[ticker]
            if 'Close' in df.columns:
                result = pd.DataFrame({ticker: df['Close'].squeeze()})
            else:
                result = df
        else:
            # Multiple tickers - concatenate
            result = pd.concat(
                {ticker: data['Close'].squeeze() for ticker, data in successful_data.items()},
                axis=1
            )
        
        # Ensure column names are just strings, not tuples
        if isinstance(result.columns, pd.MultiIndex):
            result.columns = result.columns.get_level_values(0)
            
            
        return result, failed_tickers
        
    @staticmethod
    def get_full_ticker_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch full OHLCV data for a single ticker.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            data = yf.download(
                ticker, 
                start=start_date, 
                end=end_date, 
                progress=False
            )
            
            if data.empty:
                st.warning(f"⚠️ No data available for {ticker}")
                return pd.DataFrame()
                
            # Flatten MultiIndex columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
                
            data = data.ffill(limit=5).dropna()
            return data
            
        except Exception as e:
            st.error(f"❌ Failed to load data for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def get_returns(price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate daily returns from price data.
        
        Args:
            price_data: DataFrame with price data
            
        Returns:
            DataFrame with daily returns
        """
        if price_data.empty:
            return pd.DataFrame()
        
        returns = price_data.pct_change().dropna()
        return returns
    
    @staticmethod
    @st.cache_data(ttl=86400)  # Cache sector info for 24 hours
    def get_sector_info(ticker: str) -> str:
        """
        Retrieve sector classification for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Sector name or "Unknown" if not available
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            sector = info.get('sector', 'Unknown')
            return sector if sector else 'Unknown'
        except Exception as e:
            st.warning(f"⚠️ Failed to get sector info for {ticker}: {str(e)}")
            return 'Unknown'
    
    @staticmethod
    def validate_data_sufficiency(
        data: pd.DataFrame, 
        min_days: int,
        operation: str = "Requested Operation"
    ) -> bool:
        """
        Check if sufficient data exists for analysis.
        
        Args:
            data: DataFrame to validate
            min_days: Minimum number of days required
            operation: Name of the operation requiring the data
            
        Returns:
            True if sufficient data exists, False otherwise
        """
        if data is None or data.empty:
            st.error(f"❌ No data available for {operation}")
            return False
        
        if len(data) < min_days:
            st.warning(
                f"⚠️ {operation} requires at least {min_days} days of data. "
                f"Available data: {len(data)} days"
            )
            return False
        
        return True
    
    @staticmethod
    def get_full_stock_data(
        tickers: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
        """
        Fetch full OHLCV data for given tickers.
        
        Args:
            tickers: List of stock ticker symbols
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            Tuple of (Dict mapping ticker to OHLCV DataFrame, List of failed tickers)
        """
        successful_data = {}
        failed_tickers = []
        
        for ticker in tickers:
            try:
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False
                )
                
                if data.empty:
                    failed_tickers.append(ticker)
                    st.warning(f"⚠️ لا توجد بيانات متاحة للسهم {ticker}")
                else:
                    # Handle missing data
                    data = data.ffill(limit=5)
                    data = data.dropna()
                    successful_data[ticker] = data
                    
            except Exception as e:
                failed_tickers.append(ticker)
                st.error(f"❌ فشل تحميل بيانات {ticker}: {str(e)}")
        
        return successful_data, failed_tickers
