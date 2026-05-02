import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Optional

from src.core.models import Portfolio
from src.core.session_state import SessionState
from src.features.portfolio_comparator.comparator import PortfolioComparator


def render_portfolio_creation(available_tickers: List[str]):
    """Render the UI for creating a new portfolio."""
    st.subheader("Create Portfolio")
    
    portfolios = SessionState.get_portfolios()
    if len(portfolios) >= 5:
        st.warning("⚠️ You have reached the maximum allowed limit (5 portfolios). Please delete one to add another.")
        return
        
    suggested_names = [
        "Tech Portfolio", 
        "Balanced Portfolio", 
        "High Risk Portfolio", 
        "Safe Portfolio", 
        "Diversified Portfolio",
        "Growth Portfolio",
        "Custom Portfolio 1",
        "Custom Portfolio 2"
    ]
    
    # Use text_input so users can specify custom names instead of being forced into selectbox
    name_option = st.selectbox("Portfolio Name", options=["Enter Custom Name..."] + suggested_names)
    
    if name_option == "Enter Custom Name...":
        name = st.text_input("Custom Portfolio Name*")
    else:
        name = name_option
        
    selected_tickers = st.multiselect("Select Stocks*", options=available_tickers)
    
    weights = {}
    if selected_tickers:
        st.markdown("**Weights Assignment**")
        cols = st.columns(len(selected_tickers))
        # Even distribution by default
        default_weight = 1.0 / len(selected_tickers)
        
        # Create a unique key hash based on the current selection of tickers
        # This forces Streamlit to recreate the number_input widgets with fresh default weights
        # whenever the user adds or removes a stock, preventing stale states.
        selection_hash = hash(tuple(sorted(selected_tickers)))
        
        for i, ticker in enumerate(selected_tickers):
            with cols[i]:
                weight = st.number_input(
                    f"Weight {ticker}", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=float(default_weight), 
                    step=0.05,
                    format="%.2f",
                    key=f"weight_input_{ticker}_{selection_hash}"
                )
                weights[ticker] = weight
                
    if st.button("Add Portfolio", type="primary"):
        if not name:
            st.error("❌ Please enter a portfolio name.")
        elif name in portfolios:
            st.error("❌ A portfolio with this name already exists. Please choose a different name.")
        elif not selected_tickers:
            st.error("❌ Please select at least one stock.")
        else:
            total_weight = sum(weights.values())
            if abs(total_weight - 1.0) > 0.05:
                st.error(f"❌ The sum of weights must be approximately 1.0 (Current sum: {total_weight:.2f})")
            else:
                # Normalize weights to exactly 1.0 if they are slightly off (e.g. 0.33 + 0.33 + 0.33 = 0.99)
                normalized_weights = {k: v / total_weight for k, v in weights.items()}
                portfolio = Portfolio(name=name, stocks=normalized_weights)
                SessionState.add_portfolio(name, portfolio)
                st.success(f"✅ Portfolio '{name}' has been added successfully!")


def render_portfolio_management():
    """Render UI for managing existing portfolios (view, delete)."""
    st.subheader("Manage Portfolios")
    
    portfolios = SessionState.get_portfolios()
    
    if not portfolios:
        st.info("No portfolios have been added yet.")
        return
        
    for name, p in portfolios.items():
        with st.expander(f"💼 {name}"):
            st.write("**Stocks and Weights:**")
            for ticker, weight in p.stocks.items():
                st.write(f"- {ticker}: {weight*100:.1f}%")
            
            if st.button(f"Delete '{name}'", key=f"delete_{name}"):
                SessionState.remove_portfolio(name)
                st.rerun()


def plot_comparison(returns_dict: Dict[str, pd.Series]) -> go.Figure:
    """Generate a Plotly line chart for cumulative returns."""
    fig = go.Figure()
    
    for name, returns in returns_dict.items():
        if returns.empty:
            continue
        cumulative = (1 + returns).cumprod() - 1
        fig.add_trace(go.Scatter(
            x=cumulative.index, 
            y=cumulative.values, 
            mode='lines', 
            name=name
        ))
        
    fig.update_layout(
        title="Cumulative Returns Comparison",
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        hovermode="x unified",
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def plot_drawdowns(returns_dict: Dict[str, pd.Series]) -> go.Figure:
    """Generate a Plotly chart for drawdowns."""
    fig = go.Figure()
    
    for name, returns in returns_dict.items():
        if returns.empty:
            continue
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        
        fig.add_trace(go.Scatter(
            x=drawdown.index, 
            y=drawdown.values, 
            mode='lines', 
            name=name,
            fill='tozeroy'
        ))
        
    fig.update_layout(
        title="Drawdowns Comparison",
        xaxis_title="Date",
        yaxis_title="Drawdown Ratio",
        yaxis_tickformat='.2%',
        template="plotly_dark",
        hovermode="x unified",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def plot_risk_return_scatter(metrics_df: pd.DataFrame) -> go.Figure:
    """Generate a scatter plot for risk vs return."""
    fig = px.scatter(
        metrics_df,
        x="Volatility",
        y="Annualized Return",
        text=metrics_df.index,
        color="Sharpe Ratio",
        color_continuous_scale="Viridis",
        title="Risk vs Return"
    )
    fig.update_traces(textposition='top center', marker=dict(size=12))
    fig.update_layout(
        xaxis_title="Volatility",
        yaxis_title="Annual Return",
        xaxis_tickformat='.2%',
        yaxis_tickformat='.2%',
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def render_portfolio_comparison(stock_returns: pd.DataFrame, market_returns: Optional[pd.Series] = None):
    """Render the comparison charts and metrics table for portfolios."""
    portfolios = SessionState.get_portfolios()
    
    if not portfolios:
        st.warning("⚠️ No portfolios to compare. Please create a portfolio first.")
        return
        
    if stock_returns.empty:
        st.warning("⚠️ Stock data is unavailable. Cannot perform comparison.")
        return
        
    comparator = PortfolioComparator()
    
    portfolio_returns = {}
    metrics_list = []
    names = []
    
    for name, p in portfolios.items():
        # Calculate daily returns for this portfolio
        p_returns = comparator.calculate_portfolio_returns(p, stock_returns)
        if p_returns.empty:
            continue
            
        portfolio_returns[name] = p_returns
        
        # Calculate metrics
        metrics = comparator.calculate_metrics(p_returns, market_returns)
        metrics_list.append(metrics)
        names.append(name)
        
    if not metrics_list:
        st.warning("⚠️ Not enough data to calculate metrics for the selected portfolios.")
        return
        
    # Create Metrics DataFrame
    metrics_df = pd.DataFrame(metrics_list, index=names)
    
    # Identify Best Portfolio
    best_portfolio = comparator.get_best_portfolio(portfolio_returns)
    if best_portfolio:
        st.success(f"🏆 The best portfolio based on Sharpe Ratio is: **{best_portfolio}**")
        
    # Check for High Risk Portfolios
    for name in names:
        var = metrics_df.loc[name, "VaR (95%)"]
        if var < -0.05:  # More than 5% daily VaR is high risk
            st.warning(f"⚠️ Warning: Portfolio '{name}' has very high risk (Value at Risk = {var*100:.1f}%)")

    # Display Charts
    st.plotly_chart(plot_comparison(portfolio_returns), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_risk_return_scatter(metrics_df), use_container_width=True)
    with col2:
        st.plotly_chart(plot_drawdowns(portfolio_returns), use_container_width=True)
        
    # Display Metrics Table
    st.subheader("📋 Detailed Metrics Comparison Table")
    st.info("💡 **Quick Guide:** `Annualized Return` is the expected yearly profit, while `Volatility` reflects the risk (price fluctuation). `Sharpe Ratio` measures how smart the portfolio is (higher returns with lower risk), and `VaR` shows the worst possible daily loss with 95% confidence.")
    
    # Format the dataframe for display
    display_df = metrics_df.copy()
    format_pct = lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "N/A"
    
    for col in display_df.columns:
        if col in ["Sharpe Ratio", "Beta"]:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
        else:
            display_df[col] = display_df[col].apply(format_pct)
            
    st.dataframe(display_df, use_container_width=True)
