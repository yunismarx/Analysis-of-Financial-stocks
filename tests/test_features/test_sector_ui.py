import pytest
import numpy as np
import plotly.graph_objects as go
from src.features.sector_analyzer.ui import (
    group_small_sectors,
    plot_sector_distribution,
    plot_sector_performance,
    plot_correlation_heatmap
)
from src.core.models import SectorAllocation

class TestSectorUI:
    
    def test_group_small_sectors(self):
        allocations = [
            SectorAllocation(sector="Tech", weight=0.8, stock_count=4),
            SectorAllocation(sector="Health", weight=0.15, stock_count=1),
            SectorAllocation(sector="Fin", weight=0.03, stock_count=1),
            SectorAllocation(sector="Energy", weight=0.015, stock_count=1),
            SectorAllocation(sector="Mat", weight=0.005, stock_count=1)
        ]
        
        grouped = group_small_sectors(allocations, threshold=0.02)
        
        # Tech (0.8), Health (0.15), Fin (0.03), Other (0.02)
        assert len(grouped) == 4
        
        # Check 'Other' category
        other = next((a for a in grouped if a.sector == "Other"), None)
        assert other is not None
        assert np.isclose(other.weight, 0.02)
        assert other.stock_count == 2
        
        # Check sorting
        assert grouped[0].sector == "Tech"
        
    def test_plot_sector_distribution(self):
        allocations = [
            SectorAllocation(sector="Tech", weight=0.8, stock_count=4),
            SectorAllocation(sector="Health", weight=0.2, stock_count=1)
        ]
        
        fig = plot_sector_distribution(allocations)
        assert isinstance(fig, go.Figure)
        
    def test_plot_sector_performance(self):
        sector_returns = {"Tech": 0.05, "Health": -0.02}
        fig = plot_sector_performance(sector_returns)
        assert isinstance(fig, go.Figure)
        
    def test_plot_correlation_heatmap(self):
        matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        sectors = ["Tech", "Health"]
        
        fig = plot_correlation_heatmap(matrix, sectors)
        assert isinstance(fig, go.Figure)
        
    def test_plot_correlation_heatmap_empty(self):
        fig = plot_correlation_heatmap(None, [])
        assert isinstance(fig, go.Figure)
