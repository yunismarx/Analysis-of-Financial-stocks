from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
import pandas as pd

@dataclass
class Portfolio:
    """Represents a stock portfolio with weights."""
    name: str
    stocks: Dict[str, float]  # mapping of ticker -> weight
    created_at: datetime = field(default_factory=datetime.now)

    def validate_weights(self, tolerance: float = 1e-4) -> bool:
        """
        Validate that portfolio weights sum to 1.0 and all weights are positive.
        
        Args:
            tolerance: Float tolerance for weight sum comparison.
            
        Returns:
            bool: True if weights are valid, False otherwise.
        """
        if not self.stocks:
            return False
            
        # Check for negative weights
        if any(w < 0 for w in self.stocks.values()):
            return False
            
        total_weight = sum(self.stocks.values())
        return abs(total_weight - 1.0) <= tolerance

    def get_tickers(self) -> List[str]:
        """
        Get all stock tickers in the portfolio.
        
        Returns:
            List of ticker symbols.
        """
        return list(self.stocks.keys())


@dataclass
class TechnicalIndicatorData:
    """Container for technical indicator results."""
    ticker: str
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_hist: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None


@dataclass
class SectorAllocation:
    """Represents the allocation of a specific sector in the portfolio."""
    sector: str
    weight: float
    stock_count: int


@dataclass
class SectorAnalysisResult:
    """Container for sector analysis results."""
    allocations: List[SectorAllocation]
    sector_returns: Dict[str, float]
    correlation_matrix: Optional[pd.DataFrame] = None


@dataclass
class PredictionResult:
    """Container for machine learning prediction results."""
    ticker: str
    horizon: int
    predicted_prices: List[float]
    lower_bound: List[float]
    upper_bound: List[float]
    model_name: str
    rmse: Optional[float] = None
    mae: Optional[float] = None
    mape: Optional[float] = None
    r2: Optional[float] = None


@dataclass
class ModelComparison:
    """Container for comparing multiple ML models."""
    ticker: str
    horizon: int
    results: Dict[str, PredictionResult]  # model_name -> result
    best_model: Optional[str] = None



