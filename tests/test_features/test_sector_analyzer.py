import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from src.core.models import Portfolio
from src.features.sector_analyzer.analyzer import SectorAnalyzer

class TestSectorAnalyzer:
    
    def setup_method(self):
        self.portfolio = Portfolio(
            name="Test",
            stocks={"AAPL": 0.5, "MSFT": 0.3, "JNJ": 0.2}
        )
        self.mapping = {
            "AAPL": "Technology",
            "MSFT": "Technology",
            "JNJ": "Healthcare"
        }
        
    @patch('src.features.sector_analyzer.analyzer.DataManager.get_sector_info')
    def test_get_sector_classification(self, mock_get_info):
        # Setup mock to return specific sectors
        def side_effect(ticker):
            if ticker == "AAPL": return "Technology"
            if ticker == "JNJ": return "Healthcare"
            return "Unknown"
            
        mock_get_info.side_effect = side_effect
        
        # Test without cache
        tickers = ["AAPL", "JNJ", "UNKNOWN"]
        mapping = SectorAnalyzer.get_sector_classification(tickers)
        
        assert mapping["AAPL"] == "Technology"
        assert mapping["JNJ"] == "Healthcare"
        assert mapping["UNKNOWN"] == "Unknown"
        
        # Verify it was cached
        from src.core.session_state import SessionState
        cache = SessionState.get_sector_cache()
        assert "AAPL" in cache
        
    def test_calculate_sector_allocation(self):
        allocations = SectorAnalyzer.calculate_sector_allocation(self.portfolio, self.mapping)
        
        assert len(allocations) == 2
        
        # Should be sorted by weight descending
        assert allocations[0].sector == "Technology"
        assert allocations[0].weight == 0.8
        assert allocations[0].stock_count == 2
        
        assert allocations[1].sector == "Healthcare"
        assert allocations[1].weight == 0.2
        assert allocations[1].stock_count == 1
        
    def test_calculate_sector_performance(self):
        # Create daily returns for 2 days
        stock_returns = pd.DataFrame({
            "AAPL": [0.01, 0.02], # total return ~ 0.0302
            "MSFT": [0.00, 0.01], # total return = 0.01
            "JNJ": [-0.01, -0.01] # total return ~ -0.0199
        })
        
        perf = SectorAnalyzer.calculate_sector_performance(self.portfolio, stock_returns, self.mapping)
        
        assert "Technology" in perf
        assert "Healthcare" in perf
        
        # Technology = 0.5 * AAPL_ret + 0.3 * MSFT_ret
        aapl_ret = (1.01 * 1.02) - 1.0
        msft_ret = (1.00 * 1.01) - 1.0
        expected_tech = 0.5 * aapl_ret + 0.3 * msft_ret
        
        assert np.isclose(perf["Technology"], expected_tech)
        
    def test_calculate_sector_correlation(self):
        stock_returns = pd.DataFrame({
            "AAPL": [0.01, 0.02, -0.01],
            "MSFT": [0.015, 0.01, -0.005],
            "JNJ": [-0.01, 0.00, 0.01]
        })
        
        corr = SectorAnalyzer.calculate_sector_correlation(stock_returns, self.mapping)
        
        assert corr is not None
        assert corr.shape == (2, 2)
        assert np.allclose(np.diag(corr.values), 1.0)
        
    def test_calculate_sector_correlation_insufficient_data(self):
        # Only 1 sector
        mapping = {"AAPL": "Tech", "MSFT": "Tech"}
        stock_returns = pd.DataFrame({
            "AAPL": [0.01, 0.02],
            "MSFT": [0.015, 0.01]
        })
        
        corr = SectorAnalyzer.calculate_sector_correlation(stock_returns, mapping)
        assert corr.shape == (1, 1)
        assert corr.values[0, 0] == 1.0
        
    def test_analyze_portfolio_integration(self):
        # We can test the whole pipeline
        # Mock the get_sector_classification to avoid API calls
        with patch.object(SectorAnalyzer, 'get_sector_classification', return_value=self.mapping):
            stock_returns = pd.DataFrame({
                "AAPL": [0.01, 0.02],
                "MSFT": [0.00, 0.01],
                "JNJ": [-0.01, -0.01]
            })
            
            result = SectorAnalyzer.analyze_portfolio(self.portfolio, stock_returns)
            
            assert len(result.allocations) == 2
            assert "Technology" in result.sector_returns
            assert result.correlation_matrix is not None
