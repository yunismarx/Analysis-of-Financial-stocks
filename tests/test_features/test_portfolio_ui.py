import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from src.features.portfolio_comparator.ui import plot_comparison, plot_drawdowns, plot_risk_return_scatter
from src.core.models import Portfolio
import plotly.graph_objects as go

class TestPortfolioUI:
    
    def test_plot_comparison(self):
        returns_dict = {
            "P1": pd.Series([0.01, -0.01, 0.02]),
            "P2": pd.Series([0.00, 0.01, 0.01])
        }
        
        fig = plot_comparison(returns_dict)
        assert isinstance(fig, go.Figure)
        
        # 2 portfolios means 2 traces
        assert len(fig.data) == 2
        
    def test_plot_drawdowns(self):
        returns_dict = {
            "P1": pd.Series([0.01, -0.01, 0.02]),
        }
        
        fig = plot_drawdowns(returns_dict)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        
    def test_plot_risk_return_scatter(self):
        metrics_df = pd.DataFrame({
            "Volatility": [0.1, 0.2],
            "Annualized Return": [0.05, 0.15],
            "Sharpe Ratio": [0.5, 0.75]
        }, index=["P1", "P2"])
        
        fig = plot_risk_return_scatter(metrics_df)
        assert isinstance(fig, go.Figure)
