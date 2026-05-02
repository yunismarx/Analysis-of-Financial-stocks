"""
Session State Manager - Manages Streamlit session state
"""
import streamlit as st
from datetime import datetime
from typing import Dict, List, Any, Optional


class SessionState:
    """
    Manages application state across Streamlit reruns.
    Provides centralized access to session state variables.
    """
    
    @staticmethod
    def initialize():
        """Initialize session state with default values"""
        
        # Portfolio configurations
        if 'portfolios' not in st.session_state:
            from src.core.models import Portfolio
            import json
            import os
            
            default_portfolios = {
                "Tech Giants": Portfolio(name="Tech Giants", stocks={"AAPL": 0.6, "MSFT": 0.4}),
                "Diversified": Portfolio(name="Diversified", stocks={"AAPL": 0.33, "MSFT": 0.33, "GOOGL": 0.34}),
                "Balanced": Portfolio(name="Balanced", stocks={"MSFT": 0.4, "GOOGL": 0.4, "AMZN": 0.2})
            }
            
            # Load from JSON if exists
            try:
                if os.path.exists("portfolios.json"):
                    with open("portfolios.json", "r") as f:
                        data = json.load(f)
                        for name, p_data in data.items():
                            from datetime import datetime
                            created_at = datetime.fromisoformat(p_data.get('created_at', datetime.now().isoformat()))
                            default_portfolios[name] = Portfolio(
                                name=p_data['name'],
                                stocks=p_data['stocks'],
                                created_at=created_at
                            )
            except Exception as e:
                print(f"Error loading portfolios: {e}")
                
            st.session_state.portfolios = default_portfolios
        
        # Technical indicator configurations
        if 'indicator_config' not in st.session_state:
            st.session_state.indicator_config = {
                'rsi_period': 14,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'ma_periods': [20, 50, 200]
            }
        
        # Selected tickers and date range
        if 'selected_tickers' not in st.session_state:
            st.session_state.selected_tickers = []
        
        if 'date_range' not in st.session_state:
            st.session_state.date_range = None
        
        # Cached data
        if 'cached_data' not in st.session_state:
            st.session_state.cached_data = {}
        
        # Trained ML models
        if 'trained_models' not in st.session_state:
            st.session_state.trained_models = {}
        
        # Active feature tab
        if 'active_tab' not in st.session_state:
            st.session_state.active_tab = 'overview'
        
        # Sector cache
        if 'sector_cache' not in st.session_state:
            st.session_state.sector_cache = {}
    
    @staticmethod
    def get_portfolios() -> Dict[str, Any]:
        """Get all portfolios"""
        return st.session_state.get('portfolios', {})
    
    @staticmethod
    def _save_portfolios_to_disk() -> None:
        """Save current portfolios to disk"""
        try:
            import json
            portfolios = st.session_state.get('portfolios', {})
            data = {}
            for name, p in portfolios.items():
                # Skip saving default ones if we want, but saving all is fine
                data[name] = {
                    'name': p.name,
                    'stocks': p.stocks,
                    'created_at': p.created_at.isoformat()
                }
            with open("portfolios.json", "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Error saving portfolios: {e}")

    @staticmethod
    def add_portfolio(name: str, portfolio: Any) -> None:
        """Add or update a portfolio"""
        if 'portfolios' not in st.session_state:
            st.session_state.portfolios = {}
        st.session_state.portfolios[name] = portfolio
        SessionState._save_portfolios_to_disk()
    
    @staticmethod
    def remove_portfolio(name: str) -> bool:
        """Remove a portfolio by name"""
        if 'portfolios' in st.session_state and name in st.session_state.portfolios:
            del st.session_state.portfolios[name]
            SessionState._save_portfolios_to_disk()
            return True
        return False
    
    @staticmethod
    def get_indicator_config() -> Dict[str, Any]:
        """Get technical indicator configuration"""
        return st.session_state.get('indicator_config', {
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'ma_periods': [20, 50, 200]
        })
    
    @staticmethod
    def update_indicator_config(config: Dict[str, Any]) -> None:
        """Update technical indicator configuration"""
        if 'indicator_config' not in st.session_state:
            st.session_state.indicator_config = {}
        st.session_state.indicator_config.update(config)
    
    @staticmethod
    def get_trained_model(model_name: str) -> Optional[Any]:
        """Get a trained model by name"""
        return st.session_state.get('trained_models', {}).get(model_name)
    
    @staticmethod
    def save_trained_model(model_name: str, model: Any) -> None:
        """Save a trained model"""
        if 'trained_models' not in st.session_state:
            st.session_state.trained_models = {}
        st.session_state.trained_models[model_name] = model
    
    @staticmethod
    def clear_models() -> None:
        """Clear all trained models"""
        st.session_state.trained_models = {}
    
    @staticmethod
    def get_sector_cache() -> Dict[str, str]:
        """Get sector classification cache"""
        return st.session_state.get('sector_cache', {})
    
    @staticmethod
    def cache_sector(ticker: str, sector: str) -> None:
        """Cache sector classification for a ticker"""
        if 'sector_cache' not in st.session_state:
            st.session_state.sector_cache = {}
        st.session_state.sector_cache[ticker] = sector
    
    @staticmethod
    def reset_all() -> None:
        """Reset all session state (useful for testing)"""
        keys_to_keep = []  # Add any keys that should persist
        keys_to_delete = [key for key in st.session_state.keys() if key not in keys_to_keep]
        for key in keys_to_delete:
            del st.session_state[key]
        SessionState.initialize()
