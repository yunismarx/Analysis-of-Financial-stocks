"""
Technical Indicators UI - Main interface for technical analysis
"""
import streamlit as st
import pandas as pd
from typing import List
from src.core import DataManager, ValidationService, SessionState
from .indicators import TechnicalIndicators
from .config_ui import IndicatorConfigUI


class TechnicalIndicatorsUI:
    """
    Main UI component for Technical Indicators feature.
    """
    
    @staticmethod
    def render(tickers: List[str], start_date, end_date):
        """
        Render the complete Technical Indicators UI.
        
        Args:
            tickers: List of stock ticker symbols
            start_date: Start date for data
            end_date: End date for data
        """
        st.header("📊 Technical Indicators")
        
        # Display info
        IndicatorConfigUI.display_indicator_info()
        
        # Validate inputs
        if not tickers:
            st.warning("⚠️ Please select at least one stock from the sidebar.")
            return
        
        # Get configuration
        config = SessionState.get_indicator_config()
        
        # Fetch data
        with st.spinner("Loading data..."):
            price_data, failed_tickers = DataManager.get_stock_data(
                tickers, start_date, end_date
            )
        
        if price_data.empty:
            st.error("❌ Failed to load data. Please check ticker symbols and your internet connection.")
            return
        
        # Select ticker to analyze
        if len(tickers) > 1:
            selected_ticker = st.selectbox(
                "Select Stock to Analyze",
                [t for t in tickers if t not in failed_tickers],
                help="Choose the stock to display its technical indicators"
            )
        else:
            selected_ticker = tickers[0]
        
        # Get data for selected ticker
        if selected_ticker in price_data.columns:
            ticker_data = price_data[selected_ticker]
        elif len(price_data.columns) == 1:
            ticker_data = price_data.iloc[:, 0]
        else:
            ticker_data = price_data.iloc[:, 0]
        
        # Ensure ticker_data is a Series
        if isinstance(ticker_data, pd.DataFrame):
            ticker_data = ticker_data.iloc[:, 0]
        
        # Validate data sufficiency
        ticker_df = ticker_data.to_frame() if isinstance(ticker_data, pd.Series) else ticker_data
            
        if not DataManager.validate_data_sufficiency(
            ticker_df,
            max(config['rsi_period'], config['macd_slow'], max(config['ma_periods'])),
            "Technical Indicators"
        ):
            return
        
        # Create TechnicalIndicators instance with Series
        ti = TechnicalIndicators(ticker_data)
        
        # Calculate indicators
        with st.spinner("Calculating indicators..."):
            rsi = ti.calculate_rsi(period=config['rsi_period'])
            macd_line, signal_line, histogram = ti.calculate_macd(
                fast_period=config['macd_fast'],
                slow_period=config['macd_slow'],
                signal_period=config['macd_signal']
            )
            moving_averages = ti.calculate_moving_averages(periods=config['ma_periods'])
        
        # Display current values
        col1, col2, col3 = st.columns(3)
        
        with col1:
            current_rsi = float(rsi.iloc[-1]) if not rsi.empty else 0
            rsi_status = "🔴 Overbought" if current_rsi > 70 else "🟢 Oversold" if current_rsi < 30 else "⚪ Neutral"
            st.metric(
                "Current RSI",
                f"{current_rsi:.2f}",
                delta=rsi_status
            )
        
        with col2:
            current_macd = float(macd_line.iloc[-1]) if not macd_line.empty else 0
            current_signal = float(signal_line.iloc[-1]) if not signal_line.empty else 0
            macd_status = "Bullish" if current_macd > current_signal else "Bearish"
            st.metric(
                "MACD",
                f"{current_macd:.4f}",
                delta=macd_status
            )
        
        with col3:
            if isinstance(ticker_data, pd.Series):
                current_price = float(ticker_data.iloc[-1]) if not ticker_data.empty else 0
            else:
                current_price = float(ticker_data.iloc[-1, 0]) if not ticker_data.empty else 0
            
            ma_20 = float(moving_averages[20].iloc[-1]) if 20 in moving_averages and not moving_averages[20].empty else current_price
            price_vs_ma = "Above MA20" if current_price > ma_20 else "Below MA20"
            st.metric(
                "Current Price",
                f"${current_price:.2f}",
                delta=price_vs_ma
            )
        
        # Display charts
        st.subheader("📈 Charts")
        
        # Price with Moving Averages
        # Ensure ticker_data is a Series
        if isinstance(ticker_data, pd.DataFrame):
            price_series = ticker_data.iloc[:, 0]
        else:
            price_series = ticker_data
            
        st.plotly_chart(
            ti.plot_price_with_ma(selected_ticker, price_series, moving_averages),
            use_container_width=True
        )
        
        # RSI Chart
        st.plotly_chart(
            ti.plot_rsi(selected_ticker, rsi),
            use_container_width=True
        )
        
        # MACD Chart
        st.plotly_chart(
            ti.plot_macd(selected_ticker, macd_line, signal_line, histogram),
            use_container_width=True
        )
        
        # Trading signals
        st.subheader("🎯 Trading Signals")
        
        signals = []
        
        # RSI signals
        if ti.is_overbought(rsi):
            signals.append("🔴 **RSI is Overbought** - It might be a good time to sell.")
        elif ti.is_oversold(rsi):
            signals.append("🟢 **RSI is Oversold** - It might be a good time to buy.")
        
        # MACD signals
        if not macd_line.empty and not signal_line.empty:
            if macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]:
                signals.append("🟢 **MACD Bullish Crossover** - Potential buy signal.")
            elif macd_line.iloc[-1] < signal_line.iloc[-1] and macd_line.iloc[-2] >= signal_line.iloc[-2]:
                signals.append("🔴 **MACD Bearish Crossover** - Potential sell signal.")
        
        # MA signals
        if 50 in moving_averages and 200 in moving_averages:
            ma_50 = moving_averages[50].iloc[-1]
            ma_200 = moving_averages[200].iloc[-1]
            
            if ma_50 > ma_200:
                signals.append("🟢 **Golden Cross** - MA50 is above MA200 (Uptrend).")
            elif ma_50 < ma_200:
                signals.append("🔴 **Death Cross** - MA50 is below MA200 (Downtrend).")
        
        if signals:
            for signal in signals:
                st.info(signal)
        else:
            st.info("⚪ No clear trading signals at the moment.")
        
        # Data table
        with st.expander("📋 View Raw Data"):
            # Create a clean dataframe for display
            display_data = pd.DataFrame({
                'RSI': rsi,
                'MACD': macd_line,
                'Signal': signal_line,
                'Histogram': histogram
            })
            
            # Add MA columns
            for period in sorted(moving_averages.keys()):
                display_data[f'MA{period}'] = moving_averages[period]
            
            # Format and display last 20 rows
            st.dataframe(
                display_data.tail(20).style.format("{:.4f}"),
                use_container_width=True
            )
