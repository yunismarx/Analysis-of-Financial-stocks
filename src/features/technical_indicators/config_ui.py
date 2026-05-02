"""
Technical Indicators Configuration UI
"""
import streamlit as st
from src.core import ValidationService, SessionState
from typing import Dict, Any


class IndicatorConfigUI:
    """
    Provides UI components for configuring technical indicators.
    """
    
    @staticmethod
    def render_config_sidebar() -> Dict[str, Any]:
        """
        Render configuration UI in sidebar and return configuration.
        
        Returns:
            Dictionary with indicator configuration
        """
        st.sidebar.subheader("⚙️ Technical Indicators Settings")
        
        # Initialize session state
        SessionState.initialize()
        current_config = SessionState.get_indicator_config()
        
        # RSI Configuration
        st.sidebar.markdown("**Relative Strength Index (RSI)**")
        rsi_period = st.sidebar.slider(
            "RSI Period",
            min_value=5,
            max_value=30,
            value=current_config.get('rsi_period', 14),
            help="Number of days used in RSI calculation (Default: 14)"
        )
        
        # MACD Configuration
        st.sidebar.markdown("**MACD**")
        macd_fast = st.sidebar.slider(
            "Fast Period",
            min_value=8,
            max_value=20,
            value=current_config.get('macd_fast', 12),
            help="Fast moving average period (Default: 12)"
        )
        
        macd_slow = st.sidebar.slider(
            "Slow Period",
            min_value=20,
            max_value=35,
            value=current_config.get('macd_slow', 26),
            help="Slow moving average period (Default: 26)"
        )
        
        macd_signal = st.sidebar.slider(
            "Signal Period",
            min_value=5,
            max_value=15,
            value=current_config.get('macd_signal', 9),
            help="Signal line period (Default: 9)"
        )
        
        # Moving Averages Configuration
        st.sidebar.markdown("**Moving Averages**")
        
        use_custom_ma = st.sidebar.checkbox(
            "Use Custom Periods",
            value=False,
            help="Enable to enter custom periods for moving averages"
        )
        
        if use_custom_ma:
            ma_input = st.sidebar.text_input(
                "Periods (comma-separated)",
                value="20,50,200",
                help="Enter periods separated by commas (e.g. 20,50,200)"
            )
            
            try:
                ma_periods = [int(p.strip()) for p in ma_input.split(',') if p.strip()]
                
                # Validate periods
                for period in ma_periods:
                    if period < 1 or period > 500:
                        st.sidebar.error(f"❌ Period {period} is out of bounds (1-500)")
                        ma_periods = [20, 50, 200]
                        break
            except ValueError:
                st.sidebar.error("❌ Invalid format. Using default values.")
                ma_periods = [20, 50, 200]
        else:
            ma_periods = [20, 50, 200]
        
        # Validate configuration
        config = {
            'rsi_period': rsi_period,
            'macd_fast': macd_fast,
            'macd_slow': macd_slow,
            'macd_signal': macd_signal,
            'ma_periods': ma_periods
        }
        
        # Validate MACD periods
        if macd_fast >= macd_slow:
            st.sidebar.error("❌ Fast period must be less than the slow period")
            config['macd_fast'] = 12
            config['macd_slow'] = 26
        
        # Save to session state
        SessionState.update_indicator_config(config)
        
        return config
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """
        Validate indicator configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Validate RSI period
            ValidationService.validate_indicator_period(
                config['rsi_period'],
                5, 30,
                "RSI"
            )
            
            # Validate MACD periods
            ValidationService.validate_indicator_period(
                config['macd_fast'],
                8, 20,
                "Fast MACD"
            )
            
            ValidationService.validate_indicator_period(
                config['macd_slow'],
                20, 35,
                "Slow MACD"
            )
            
            ValidationService.validate_indicator_period(
                config['macd_signal'],
                5, 15,
                "MACD Signal"
            )
            
            # Validate MACD relationship
            if config['macd_fast'] >= config['macd_slow']:
                raise ValidationService.ValidationError(
                    "Fast period must be less than the slow period"
                )
            
            # Validate MA periods
            for period in config['ma_periods']:
                if not isinstance(period, int) or period < 1 or period > 500:
                    raise ValidationService.ValidationError(
                        f"Moving average period {period} is invalid (must be between 1 and 500)"
                    )
            
            return True
            
        except Exception as e:
            st.error(f"❌ Configuration validation error: {str(e)}")
            return False
    
    @staticmethod
    def display_indicator_info():
        """Display information about technical indicators"""
        with st.expander("ℹ️ About Technical Indicators"):
            st.markdown("""
            ### Relative Strength Index (RSI)
            - **Range**: 0-100
            - **Overbought**: Above 70 (Might be time to sell)
            - **Oversold**: Below 30 (Might be time to buy)
            - **Usage**: Measures the speed and change of price movements
            
            ### MACD (Moving Average Convergence Divergence)
            - **MACD Line**: Difference between the fast and slow moving averages
            - **Signal Line**: Moving average of the MACD line
            - **Histogram**: Difference between the MACD line and the Signal line
            - **Usage**: Identifies trend momentum and strength
            
            ### Moving Averages (MA)
            - **MA 20**: Short-term trend
            - **MA 50**: Medium-term trend
            - **MA 200**: Long-term trend
            - **Usage**: Identifies trends, support, and resistance levels
            """)
