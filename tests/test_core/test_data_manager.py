import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.core.data_manager import DataManager

class TestDataManager:
    
    @patch('src.core.data_manager.yf.download')
    def test_get_stock_data_success(self, mock_download):
        # Setup mock
        mock_df = pd.DataFrame({
            'Close': [100, 101, 102]
        }, index=pd.date_range(start='2023-01-01', periods=3))
        mock_download.return_value = mock_df
        
        # Call method
        tickers = ['AAPL']
        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 3)
        result, failed = DataManager.get_stock_data(tickers, start, end)
        
        # Assertions
        assert len(failed) == 0
        assert not result.empty
        assert 'AAPL' in result.columns
        assert list(result['AAPL']) == [100, 101, 102]
        
    @patch('src.core.data_manager.yf.download')
    def test_get_stock_data_failure(self, mock_download):
        # Setup mock to return empty dataframe (failure case)
        mock_download.return_value = pd.DataFrame()
        
        # Call method
        tickers = ['INVALID']
        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 3)
        result, failed = DataManager.get_stock_data(tickers, start, end)
        
        # Assertions
        assert 'INVALID' in failed
        assert result.empty
        
    @patch('src.core.data_manager.yf.download')
    def test_get_stock_data_multiple(self, mock_download):
        # Setup mock
        def side_effect(*args, **kwargs):
            if args[0] == 'AAPL':
                return pd.DataFrame({'Close': [150, 151]}, index=pd.date_range('2023-01-01', periods=2))
            elif args[0] == 'MSFT':
                return pd.DataFrame({'Close': [250, 252]}, index=pd.date_range('2023-01-01', periods=2))
            return pd.DataFrame()
            
        mock_download.side_effect = side_effect
        
        # Call method
        tickers = ['AAPL', 'MSFT', 'INVALID']
        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 2)
        result, failed = DataManager.get_stock_data(tickers, start, end)
        
        # Assertions
        assert 'INVALID' in failed
        assert 'AAPL' in result.columns
        assert 'MSFT' in result.columns
        assert result['AAPL'].iloc[0] == 150
        assert result['MSFT'].iloc[1] == 252
        
    def test_get_returns(self):
        # Setup data
        prices = pd.DataFrame({
            'AAPL': [100.0, 101.0, 99.99],
            'MSFT': [200.0, 204.0, 199.92]
        })
        
        # Call method
        returns = DataManager.get_returns(prices)
        
        # Assertions
        assert len(returns) == 2  # One less than prices due to pct_change
        assert np.isclose(returns['AAPL'].iloc[0], 0.01) # 101/100 - 1
        assert np.isclose(returns['MSFT'].iloc[0], 0.02) # 204/200 - 1
        
    def test_get_returns_empty(self):
        returns = DataManager.get_returns(pd.DataFrame())
        assert returns.empty
        
    @patch('src.core.data_manager.yf.Ticker')
    def test_get_sector_info(self, mock_ticker):
        # Setup mock
        mock_instance = MagicMock()
        mock_instance.info = {'sector': 'Technology'}
        mock_ticker.return_value = mock_instance
        
        # Call method
        sector = DataManager.get_sector_info('AAPL')
        
        # Assertions
        assert sector == 'Technology'
        
    @patch('src.core.data_manager.yf.Ticker')
    def test_get_sector_info_failure(self, mock_ticker):
        # Setup mock to raise exception
        mock_ticker.side_effect = Exception("API Error")
        
        # Call method
        sector = DataManager.get_sector_info('INVALID')
        
        # Assertions
        assert sector == 'Unknown'
