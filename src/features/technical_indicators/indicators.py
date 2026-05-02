"""
Technical Indicators Module - RSI, MACD, Moving Averages
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Tuple, Dict, List, Optional


class TechnicalIndicators:
    """
    Calculate and visualize technical indicators for stock analysis.
    Supports RSI, MACD, and Moving Averages.
    """
    
    def __init__(self, price_data: pd.DataFrame):
        """
        Initialize TechnicalIndicators with price data.
        
        Args:
            price_data: DataFrame with price data (must have 'Close' column or be a Series)
        """
        self.price_data = price_data
    
    def calculate_rsi(self, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index using Wilder's smoothing method.
        
        RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss
        
        Args:
            period: Period for RSI calculation (default: 14)
            
        Returns:
            Series with RSI values (0-100)
        """
        # Get close prices
        if isinstance(self.price_data, pd.Series):
            close = self.price_data
        else:
            close = self.price_data['Close'] if 'Close' in self.price_data.columns else self.price_data.iloc[:, 0]
        
        # Calculate price changes
        delta = close.diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss using Wilder's smoothing
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        MACD Line = EMA(fast) - EMA(slow)
        Signal Line = EMA(MACD Line, signal_period)
        Histogram = MACD Line - Signal Line
        
        Args:
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal line EMA period (default: 9)
            
        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        # Get close prices
        if isinstance(self.price_data, pd.Series):
            close = self.price_data
        else:
            close = self.price_data['Close'] if 'Close' in self.price_data.columns else self.price_data.iloc[:, 0]
        
        # Calculate EMAs
        ema_fast = close.ewm(span=fast_period, adjust=False).mean()
        ema_slow = close.ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def calculate_moving_averages(
        self,
        periods: List[int] = [20, 50, 200]
    ) -> Dict[int, pd.Series]:
        """
        Calculate Simple Moving Averages for given periods.
        
        SMA(n) = Sum(Close[i-n+1 to i]) / n
        
        Args:
            periods: List of periods for moving averages (default: [20, 50, 200])
            
        Returns:
            Dictionary mapping period to SMA Series
        """
        # Get close prices
        if isinstance(self.price_data, pd.Series):
            close = self.price_data
        else:
            close = self.price_data['Close'] if 'Close' in self.price_data.columns else self.price_data.iloc[:, 0]
        
        moving_averages = {}
        for period in periods:
            moving_averages[period] = close.rolling(window=period).mean()
        
        return moving_averages
    
    def _calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """
        Helper method to calculate Exponential Moving Average.
        
        Args:
            data: Price data
            period: EMA period
            
        Returns:
            EMA Series
        """
        return data.ewm(span=period, adjust=False).mean()
    
    def plot_rsi(
        self,
        ticker: str,
        rsi: pd.Series,
        overbought: float = 70,
        oversold: float = 30
    ) -> go.Figure:
        """
        Create RSI visualization with overbought/oversold zones.
        
        Args:
            ticker: Stock ticker symbol
            rsi: RSI Series
            overbought: Overbought threshold (default: 70)
            oversold: Oversold threshold (default: 30)
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure()
        
        # Add RSI line
        fig.add_trace(go.Scatter(
            x=rsi.index,
            y=rsi,
            mode='lines',
            name='RSI',
            line=dict(color='#0078D4', width=2.5)
        ))
        
        # Add overbought line
        fig.add_hline(
            y=overbought,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Overbought Zone ({overbought})",
            annotation_position="right"
        )
        
        # Add oversold line
        fig.add_hline(
            y=oversold,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Oversold Zone ({oversold})",
            annotation_position="right"
        )
        
        # Add middle line
        fig.add_hline(
            y=50,
            line_dash="dot",
            line_color="gray",
            opacity=0.5
        )
        
        # Update layout
        fig.update_layout(
            title=f'Relative Strength Index (RSI) - {ticker}',
            xaxis_title='Date',
            yaxis_title='RSI',
            yaxis=dict(range=[0, 100]),
            hovermode='x unified',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        
        return fig
    
    def plot_macd(
        self,
        ticker: str,
        macd_line: pd.Series,
        signal_line: pd.Series,
        histogram: pd.Series
    ) -> go.Figure:
        """
        Create MACD visualization with line, signal, and histogram.
        
        Args:
            ticker: Stock ticker symbol
            macd_line: MACD line Series
            signal_line: Signal line Series
            histogram: Histogram Series
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure()
        
        # Add MACD line
        fig.add_trace(go.Scatter(
            x=macd_line.index,
            y=macd_line,
            mode='lines',
            name='MACD',
            line=dict(color='#0078D4', width=2.5)
        ))
        
        # Add signal line
        fig.add_trace(go.Scatter(
            x=signal_line.index,
            y=signal_line,
            mode='lines',
            name='Signal',
            line=dict(color='#FFB900', width=2.5)
        ))
        
        # Add histogram with color coding
        colors = ['green' if val >= 0 else 'red' for val in histogram]
        fig.add_trace(go.Bar(
            x=histogram.index,
            y=histogram,
            name='Histogram',
            marker_color=colors,
            opacity=0.5
        ))
        
        # Add zero line
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="gray",
            opacity=0.5
        )
        
        # Update layout
        fig.update_layout(
            title=f'MACD (Moving Average Convergence Divergence) - {ticker}',
            xaxis_title='Date',
            yaxis_title='Value',
            hovermode='x unified',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        
        return fig
    
    def plot_price_with_ma(
        self,
        ticker: str,
        price: pd.Series,
        moving_averages: Dict[int, pd.Series]
    ) -> go.Figure:
        """
        Create price chart with moving averages overlay.
        
        Args:
            ticker: Stock ticker symbol
            price: Price Series
            moving_averages: Dictionary mapping period to MA Series
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure()
        
        # Add price line
        fig.add_trace(go.Scatter(
            x=price.index,
            y=price,
            mode='lines',
            name='Price',
            line=dict(color='#FAFAFA', width=2.5)
        ))
        
        # Add moving averages with different colors
        colors = ['#0078D4', '#FFB900', '#107C10', '#E81123', '#8661C5']
        line_styles = ['solid', 'dash', 'dot', 'dashdot', 'longdash']
        
        for idx, (period, ma) in enumerate(sorted(moving_averages.items())):
            color = colors[idx % len(colors)]
            line_style = line_styles[idx % len(line_styles)]
            
            fig.add_trace(go.Scatter(
                x=ma.index,
                y=ma,
                mode='lines',
                name=f'MA {period}',
                line=dict(color=color, width=1.5, dash=line_style)
            ))
        
        # Update layout
        fig.update_layout(
            title=f'Price with Moving Averages - {ticker}',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            hovermode='x unified',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=500
        )
        
        return fig
    
    def is_overbought(self, rsi: pd.Series, threshold: float = 70) -> bool:
        """
        Check if latest RSI indicates overbought condition.
        
        Args:
            rsi: RSI Series
            threshold: Overbought threshold (default: 70)
            
        Returns:
            True if overbought, False otherwise
        """
        if rsi.empty:
            return False
        return rsi.iloc[-1] > threshold
    
    def is_oversold(self, rsi: pd.Series, threshold: float = 30) -> bool:
        """
        Check if latest RSI indicates oversold condition.
        
        Args:
            rsi: RSI Series
            threshold: Oversold threshold (default: 30)
            
        Returns:
            True if oversold, False otherwise
        """
        if rsi.empty:
            return False
        return rsi.iloc[-1] < threshold
