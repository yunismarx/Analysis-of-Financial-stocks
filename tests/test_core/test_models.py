import pytest
from datetime import datetime
from src.core.models import (
    Portfolio, 
    TechnicalIndicatorData, 
    SectorAllocation, 
    SectorAnalysisResult,
    PredictionResult,
    ModelComparison
)
import numpy as np

class TestPortfolioModel:
    
    def test_portfolio_initialization(self):
        portfolio = Portfolio(
            name="Tech Heavy",
            stocks={"AAPL": 0.6, "MSFT": 0.4}
        )
        assert portfolio.name == "Tech Heavy"
        assert len(portfolio.stocks) == 2
        assert isinstance(portfolio.created_at, datetime)
        
    def test_validate_weights_valid(self):
        portfolio = Portfolio(
            name="Valid",
            stocks={"AAPL": 0.6, "MSFT": 0.4}
        )
        assert portfolio.validate_weights() is True
        
    def test_validate_weights_invalid_sum(self):
        portfolio = Portfolio(
            name="Invalid Sum",
            stocks={"AAPL": 0.6, "MSFT": 0.5}
        )
        assert portfolio.validate_weights() is False
        
    def test_validate_weights_negative(self):
        portfolio = Portfolio(
            name="Negative Weight",
            stocks={"AAPL": 1.2, "MSFT": -0.2}
        )
        assert portfolio.validate_weights() is False
        
    def test_validate_weights_empty(self):
        portfolio = Portfolio(name="Empty", stocks={})
        assert portfolio.validate_weights() is False
        
    def test_get_tickers(self):
        portfolio = Portfolio(
            name="Test",
            stocks={"AAPL": 0.3, "MSFT": 0.3, "GOOGL": 0.4}
        )
        tickers = portfolio.get_tickers()
        assert set(tickers) == {"AAPL", "MSFT", "GOOGL"}
        assert len(tickers) == 3

class TestOtherModels:
    
    def test_technical_indicator_data(self):
        data = TechnicalIndicatorData(ticker="AAPL", rsi=65.5)
        assert data.ticker == "AAPL"
        assert data.rsi == 65.5
        assert data.macd is None
        
    def test_sector_allocation(self):
        alloc = SectorAllocation(sector="Technology", weight=0.45, stock_count=5)
        assert alloc.sector == "Technology"
        assert alloc.weight == 0.45
        assert alloc.stock_count == 5
        
    def test_sector_analysis_result(self):
        alloc1 = SectorAllocation(sector="Tech", weight=0.6, stock_count=2)
        result = SectorAnalysisResult(
            allocations=[alloc1],
            sector_returns={"Tech": 0.05}
        )
        assert len(result.allocations) == 1
        assert result.sector_returns["Tech"] == 0.05
        assert result.correlation_matrix is None
        
    def test_prediction_result(self):
        pred = PredictionResult(
            ticker="AAPL",
            horizon=5,
            predicted_prices=[150, 151, 152, 153, 154],
            lower_bound=[145]*5,
            upper_bound=[160]*5,
            model_name="LSTM"
        )
        assert pred.ticker == "AAPL"
        assert len(pred.predicted_prices) == 5
        assert pred.model_name == "LSTM"
        
    def test_model_comparison(self):
        pred = PredictionResult(
            ticker="AAPL", horizon=1, predicted_prices=[150], 
            lower_bound=[145], upper_bound=[155], model_name="LSTM"
        )
        comp = ModelComparison(
            ticker="AAPL",
            horizon=1,
            results={"LSTM": pred},
            best_model="LSTM"
        )
        assert comp.ticker == "AAPL"
        assert comp.best_model == "LSTM"
        assert "LSTM" in comp.results
