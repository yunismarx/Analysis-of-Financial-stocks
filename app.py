import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import norm

# Import new features
from src.core import SessionState
from src.core.data_manager import DataManager
from src.features.technical_indicators import TechnicalIndicatorsUI, IndicatorConfigUI
from src.features.portfolio_comparator.ui import (
    render_portfolio_creation,
    render_portfolio_management,
    render_portfolio_comparison,
)
from src.features.sector_analyzer.ui import render_sector_analysis_ui
from src.features.ml_predictor.ui import render_ml_predictor_ui
from src.features.probabilistic.ui import render_probabilistic_ui

# 1. Dashboard Settings
st.set_page_config(
    page_title="Advanced Portfolio Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Apple-Inspired Design System ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,300;0,14..32,400;0,14..32,500;0,14..32,600;0,14..32,700&display=swap');

/* ── Base ── */
.stApp {
    background-color: #000000;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', sans-serif;
    color: #f5f5f7;
}

/* ── Hide Streamlit chrome, keep toggle ── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header[data-testid="stHeader"] {
    background: transparent !important;
    visibility: visible !important;
}
[data-testid="stToolbar"]    { visibility: hidden; }
[data-testid="stDecoration"] { display: none !important; }

/* Force sidebar toggle buttons visible at all times */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapseButton"] *,
[data-testid="collapsedControl"],
[data-testid="collapsedControl"] *,
[data-testid="stExpandSidebarButton"],
[data-testid="stExpandSidebarButton"] * {
    visibility: visible !important;
    display: flex !important;
    opacity: 1 !important;
}
header[data-testid="stHeader"] > div,
header[data-testid="stHeader"] > div > div,
header[data-testid="stHeader"] > div > div > div {
    visibility: visible !important;
}
[data-testid="collapsedControl"] {
    background: rgba(255,255,255,0.06) !important;
    border-radius: 0 10px 10px 0 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-left: none !important;
    backdrop-filter: blur(20px) !important;
    transition: background 0.2s ease !important;
}
[data-testid="collapsedControl"]:hover {
    background: rgba(255,255,255,0.12) !important;
}
[data-testid="collapsedControl"] svg { fill: rgba(255,255,255,0.7) !important; }

/* ── Sidebar shell ── */
[data-testid="stSidebar"] {
    background: #1c1c1e !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
    padding: 0 !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
}

/* ── Sidebar section category labels ── */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-size: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.10em !important;
    text-transform: uppercase !important;
    color: rgba(255,255,255,0.35) !important;
    margin: 20px 0 6px 0 !important;
    padding: 0 4px !important;
}

/* ── Sidebar input labels ── */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stDateInput label {
    font-size: 11px !important;
    font-weight: 500 !important;
    color: rgba(255,255,255,0.45) !important;
    letter-spacing: 0.03em !important;
    text-transform: uppercase !important;
}

/* ── Sidebar input fields ── */
[data-testid="stSidebar"] [data-testid="stTextInput"] input,
[data-testid="stSidebar"] [data-baseweb="input"] input,
[data-testid="stSidebar"] [data-baseweb="select"] {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
    color: #f5f5f7 !important;
    font-size: 13px !important;
}

/* ── Sidebar multiselect chips ── */
[data-testid="stSidebar"] [data-baseweb="tag"] {
    background: rgba(0, 113, 227, 0.25) !important;
    border: 1px solid rgba(0, 113, 227, 0.4) !important;
    border-radius: 6px !important;
    color: #60a5fa !important;
    font-size: 11px !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] [data-baseweb="tag"] span[role="button"] {
    color: rgba(96, 165, 250, 0.7) !important;
}

/* ── Sidebar radio — Apple-style list items ── */
[data-testid="stSidebar"] [data-testid="stRadio"] {
    gap: 0 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    display: flex !important;
    flex-direction: column !important;
    gap: 2px !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    display: flex !important;
    align-items: center !important;
    padding: 9px 12px !important;
    border-radius: 9px !important;
    cursor: pointer !important;
    transition: background 0.15s ease !important;
    font-size: 14px !important;
    font-weight: 400 !important;
    color: rgba(255,255,255,0.75) !important;
    margin: 1px 0 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(255,255,255,0.06) !important;
    color: #f5f5f7 !important;
}
/* Hide the default radio circle */
[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stMarkdownContainer"] p { margin: 0 !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"] { display: none !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"]:checked + div + label,
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: rgba(0, 113, 227, 0.18) !important;
    color: #60a5fa !important;
    font-weight: 500 !important;
}

/* ── Sidebar horizontal rule ── */
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.08) !important;
    margin: 12px 0 !important;
}

/* ── Main page typography ── */
h1 { font-size: 2rem !important; font-weight: 700 !important; letter-spacing: -0.03em !important; color: #f5f5f7 !important; }
h2 { font-size: 1.35rem !important; font-weight: 600 !important; letter-spacing: -0.02em !important; color: #f5f5f7 !important; }
h3 { font-size: 1.05rem !important; font-weight: 600 !important; color: #f5f5f7 !important; }

/* ── Metric cards — Apple glass card ── */
[data-testid="metric-container"] {
    background: rgba(28, 28, 30, 0.8) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 14px !important;
    padding: 1.1rem 1.3rem !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    box-shadow: 0 2px 16px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.05) !important;
    transition: transform 0.18s ease, box-shadow 0.18s ease !important;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 28px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.07) !important;
}
[data-testid="stMetricLabel"] {
    font-size: 11px !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: rgba(255,255,255,0.4) !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    color: #f5f5f7 !important;
}
[data-testid="stMetricDelta"] { font-size: 12px !important; font-weight: 500 !important; }

/* ── Buttons — Apple-style pill ── */
.stButton > button {
    background: #0071e3 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 980px !important;      /* Apple's signature pill */
    padding: 0.55rem 1.4rem !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    letter-spacing: -0.01em !important;
    transition: all 0.2s ease !important;
    box-shadow: none !important;
}
.stButton > button:hover {
    background: #0077ed !important;
    transform: scale(1.02) !important;
    box-shadow: 0 4px 16px rgba(0,113,227,0.35) !important;
}
.stButton > button[kind="primary"] {
    background: #0071e3 !important;
}
.stButton > button[kind="secondary"] {
    background: rgba(255,255,255,0.1) !important;
    color: #f5f5f7 !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    background: rgba(28,28,30,0.7) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}
.streamlit-expanderHeader {
    font-size: 13px !important;
    font-weight: 500 !important;
    color: rgba(255,255,255,0.75) !important;
    background: transparent !important;
    padding: 0.85rem 1rem !important;
}
.streamlit-expanderHeader:hover {
    color: #f5f5f7 !important;
    background: rgba(255,255,255,0.04) !important;
}

/* ── DataFrames ── */
.stDataFrame {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
}

/* ── Info / Warning / Success boxes ── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border: none !important;
    backdrop-filter: blur(10px) !important;
}

/* ── Sliders ── */
[data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stThumbValue"] {
    color: #60a5fa !important;
    font-weight: 600 !important;
}

/* ── Selectbox / Multiselect ── */
[data-baseweb="select"] > div {
    background: rgba(28,28,30,0.9) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #f5f5f7 !important;
}

/* ── Horizontal rule in main area ── */
hr {
    border-color: rgba(255,255,255,0.06) !important;
    margin: 1.5rem 0 !important;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
SessionState.initialize()

st.title("📊 Advanced Portfolio & Risk Dashboard")
st.markdown("Smart analytics, expected value, and AI-powered future forecasting for financial portfolios.")
st.markdown("---")

# ── Sidebar Branding Header ───────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="
    padding: 28px 16px 20px 16px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 8px;
">
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:4px;">
        <div style="
            width:34px; height:34px;
            background: linear-gradient(135deg, #0071e3 0%, #34aadc 100%);
            border-radius:8px;
            display:flex; align-items:center; justify-content:center;
            font-size:18px;
            box-shadow: 0 2px 12px rgba(0,113,227,0.4);
        ">📊</div>
        <div>
            <div style="
                font-size:14px;
                font-weight:600;
                color:#f5f5f7;
                letter-spacing:-0.01em;
                line-height:1.2;
            ">Portfolio</div>
            <div style="
                font-size:11px;
                font-weight:400;
                color:rgba(255,255,255,0.35);
                letter-spacing:0.02em;
            ">Risk Dashboard</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# 2. Sidebar for stock selection
st.sidebar.markdown("""
<div style="
    font-size:10px; font-weight:600; letter-spacing:0.10em;
    text-transform:uppercase; color:rgba(255,255,255,0.35);
    padding: 4px 4px 6px 4px; margin-top:4px;
">Markets</div>
""", unsafe_allow_html=True)

# Stock list with full company names
STOCK_INFO = {
    # Technology
    "AAPL":  "🍎 Apple Inc.",
    "MSFT":  "🪟 Microsoft Corp.",
    "GOOGL": "🔍 Alphabet (Google)",
    "AMZN":  "📦 Amazon.com Inc.",
    "META":  "👓 Meta Platforms",
    "TSLA":  "⚡ Tesla Inc.",
    "NVDA":  "🎮 NVIDIA Corp.",
    "ORCL":  "🔵 Oracle Corp.",
    "ADBE":  "🎨 Adobe Inc.",
    "CRM":   "☁️ Salesforce Inc.",
    # Finance
    "JPM":   "🏦 JPMorgan Chase",
    "GS":    "💰 Goldman Sachs",
    "BAC":   "🏛️ Bank of America",
    "V":     "💳 Visa Inc.",
    "MA":    "💳 Mastercard Inc.",
    # Healthcare
    "JNJ":   "💊 Johnson & Johnson",
    "PFE":   "💉 Pfizer Inc.",
    "UNH":   "🏥 UnitedHealth Group",
    # Consumer
    "KO":    "🥤 Coca-Cola Co.",
    "MCD":   "🍟 McDonald's Corp.",
    "WMT":   "🛒 Walmart Inc.",
    # Energy
    "XOM":   "⛽ ExxonMobil Corp.",
    "CVX":   "🛢️ Chevron Corp.",
}

ALL_TICKERS = list(STOCK_INFO.keys())

tickers = st.sidebar.multiselect(
    "Select Stocks:",
    options=ALL_TICKERS,
    default=["AAPL", "MSFT"],
    format_func=lambda t: f"{t}  —  {STOCK_INFO.get(t, t)}",
)
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# Add manual refresh button
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Update Data Now", help="Fetch latest prices immediately without waiting for auto-refresh", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

# Add feature selection
st.sidebar.markdown("---")
st.sidebar.header("📊 Advanced Features")
feature_tab = st.sidebar.radio(
    "Select Feature",
    ["Probabilistic Analysis", "Technical Indicators", "Portfolio Comparator", "Sector Analyzer", "ML Predictor"],
    help="Choose the type of analysis you want to display"
)

# Technical Indicators Configuration (only show if selected)
if feature_tab == "Technical Indicators":
    config = IndicatorConfigUI.render_config_sidebar()

if len(tickers) < 1:
    st.warning("Please select at least one stock from the sidebar.")
elif feature_tab == "Technical Indicators":
    # Render Technical Indicators UI
    TechnicalIndicatorsUI.render(tickers, start_date, end_date)
elif feature_tab == "Portfolio Comparator":
    st.header("⚖️ Portfolio Comparator")
    
    # Render Creation and Management
    col1, col2 = st.columns(2)
    with col1:
        render_portfolio_creation(tickers)
    with col2:
        render_portfolio_management()
        
    st.markdown("---")
    
    # Load data for comparison
    with st.spinner("Fetching data..."):
        stock_data, failed = DataManager.get_stock_data(tickers, start_date, end_date)
        if not stock_data.empty:
            stock_returns = DataManager.get_returns(stock_data)
            
            # For simplicity, use an equal weighted index of selected stocks as the 'market' for Beta calculation
            market_returns = stock_returns.mean(axis=1)
            
            st.subheader("📊 Portfolio Comparison")
            render_portfolio_comparison(stock_returns, market_returns)
        else:
            st.error("Not enough data found to perform comparison.")
elif feature_tab == "Sector Analyzer":
    st.header("🏢 Sector Analyzer")
    
    portfolios = SessionState.get_portfolios()
    if not portfolios:
        st.warning("⚠️ No portfolios available to analyze. Please go to 'Portfolio Comparator' and create one first.")
    else:
        # Select portfolio to analyze
        selected_portfolio_name = st.selectbox("Select portfolio to analyze sectors:", options=list(portfolios.keys()))
        selected_portfolio = portfolios[selected_portfolio_name]
        
        # Load data for analysis
        with st.spinner("Fetching data..."):
            portfolio_tickers = selected_portfolio.get_tickers()
            stock_data, failed = DataManager.get_stock_data(portfolio_tickers, start_date, end_date)
            
            if not stock_data.empty:
                stock_returns = DataManager.get_returns(stock_data)
                render_sector_analysis_ui(selected_portfolio, stock_returns)
            else:
                st.error("Not enough data to analyze sectors.")
elif feature_tab == "ML Predictor":
    portfolios = SessionState.get_portfolios()
    if not portfolios:
        st.warning("⚠️ Please create a portfolio first in 'Portfolio Comparator' to use the predictor.")
    else:
        selected_portfolio_name = st.selectbox("Select portfolio for predictions:", options=list(portfolios.keys()))
        selected_portfolio = portfolios[selected_portfolio_name]
        
        # We pass dates back 3 years to ensure we have enough data (min 100 days)
        ml_start_date = pd.Timestamp.today() - pd.Timedelta(days=365*3)
        render_ml_predictor_ui(selected_portfolio, ml_start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
else:
    # Enhanced Probabilistic Analysis
    if len(tickers) < 2:
        st.warning("⚠️ Please select at least two stocks from the sidebar to enable the full Probabilistic Analysis.")
    else:
        with st.spinner("Fetching market data..."):
            stock_data, failed = DataManager.get_stock_data(tickers, start_date, end_date)

        if stock_data.empty:
            st.error("❌ Could not retrieve data. Please check your internet connection and selected tickers.")
        else:
            returns = DataManager.get_returns(stock_data)
            if returns.empty or len(returns) < 30:
                st.warning("⚠️ Not enough historical data (minimum 30 days required). Try a wider date range.")
            else:
                render_probabilistic_ui(tickers, returns)