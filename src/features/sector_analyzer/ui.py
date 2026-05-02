import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict

from src.core.models import Portfolio, SectorAllocation, SectorAnalysisResult
from src.features.sector_analyzer.analyzer import SectorAnalyzer

def group_small_sectors(allocations: List[SectorAllocation], threshold: float = 0.02) -> List[SectorAllocation]:
    """
    Group sectors with weight less than the threshold into an 'Other' category.
    Sorts the result descending by weight.
    """
    main_allocs = []
    other_weight = 0.0
    other_count = 0
    
    for alloc in allocations:
        if alloc.weight < threshold:
            other_weight += alloc.weight
            other_count += alloc.stock_count
        else:
            main_allocs.append(alloc)
            
    if other_weight > 0:
        main_allocs.append(SectorAllocation(
            sector="Other",
            weight=other_weight,
            stock_count=other_count
        ))
        
    main_allocs.sort(key=lambda x: x.weight, reverse=True)
    return main_allocs

def plot_sector_distribution(allocations: List[SectorAllocation]) -> go.Figure:
    """Generate a pie chart for sector distribution."""
    grouped_allocs = group_small_sectors(allocations)
    
    labels = [f"{a.sector} ({a.stock_count} stocks)" for a in grouped_allocs]
    values = [a.weight for a in grouped_allocs]
    
    fig = px.pie(
        names=labels,
        values=values
    )
    fig.update_layout(
        title='Asset Allocation by Sector',
        height=400,
        margin=dict(t=50, b=0, l=0, r=0),
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    fig.update_traces(
        textposition='inside', 
        textinfo='percent+label',
        hovertemplate="%{label}<br>Weight: %{percent}<extra></extra>"
    )
    
    return fig

def plot_sector_performance(sector_returns: Dict[str, float]) -> go.Figure:
    """Generate a bar chart for sector performance with color coding."""
    df = pd.DataFrame([
        {"Sector": sector, "Return": ret} 
        for sector, ret in sector_returns.items()
    ])
    
    if df.empty:
        return go.Figure()
        
    # Sort descending
    df = df.sort_values(by="Return", ascending=True)
    
    # Color coding: Green for positive, Red for negative
    df['Color'] = df['Return'].apply(lambda x: 'green' if x >= 0 else 'red')
    
    fig = go.Figure(go.Bar(
        x=df['Return'],
        y=df['Sector'],
        orientation='h',
        marker_color=df['Color'],
        text=df['Return'].apply(lambda x: f"{x*100:.2f}%"),
        textposition='auto'
    ))
    
    fig.update_layout(
        title='Average Daily Return by Sector',
        xaxis_title='Sector',
        yaxis_title='Return (%)',
        height=400,
        showlegend=False,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def plot_correlation_heatmap(corr_matrix: np.ndarray, sectors: List[str]) -> go.Figure:
    """Generate a heatmap for sector correlation."""
    if corr_matrix is None or len(sectors) == 0:
        return go.Figure()
        
    fig = px.imshow(
        corr_matrix,
        x=sectors,
        y=sectors,
        color_continuous_scale="RdBu_r"
    )
    fig.update_layout(
        title='Correlation Heatmap between Sectors',
        height=500,
        xaxis_title="",
        yaxis_title="",
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # Add text annotations
    for i in range(len(sectors)):
        for j in range(len(sectors)):
            fig.add_annotation(
                x=j, y=i,
                text=f"{corr_matrix[i, j]:.2f}",
                showarrow=False,
                font=dict(color="white" if abs(corr_matrix[i, j]) > 0.5 else "black")
            )
            
    fig.update_layout(template="plotly_dark")
    return fig

def render_sector_analysis_ui(portfolio: Portfolio, stock_returns: pd.DataFrame):
    """Main UI render function for Sector Analyzer."""
    st.subheader(f"📊 Sector Analysis for Portfolio: {portfolio.name}")
    
    with st.spinner("Analyzing sectors..."):
        result = SectorAnalyzer.analyze_portfolio(portfolio, stock_returns)
        
    if not result.allocations:
        st.warning("Not enough data to analyze sectors for this portfolio.")
        return
        
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(plot_sector_distribution(result.allocations), use_container_width=True)
        
    with col2:
        st.plotly_chart(plot_sector_performance(result.sector_returns), use_container_width=True)
        
    st.markdown("---")
    st.subheader("🔗 Sector Correlation")
    st.markdown("This analysis helps you understand if sectors move together. A value close to `1` means a strong positive correlation, and close to `-1` means an inverse correlation, which helps in diversifying risks.")
    
    if result.correlation_matrix is not None:
        sectors = result.correlation_matrix.columns.tolist()
        st.plotly_chart(plot_correlation_heatmap(result.correlation_matrix.values, sectors), use_container_width=True)
    else:
        st.info("Not enough sectors to calculate a correlation matrix.")
