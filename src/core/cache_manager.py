"""
Cache Manager - Manages caching strategies for the application
"""
import streamlit as st
from typing import Any, Callable, Optional
from functools import wraps


class CacheManager:
    """
    Manages caching strategies and cache invalidation for the application.
    Provides utilities for working with Streamlit's caching system.
    """
    
    @staticmethod
    def clear_all_caches():
        """Clear all cached data."""
        st.cache_data.clear()
        st.success("✅ All cached data has been cleared")
    
    @staticmethod
    def clear_data_cache():
        """Clear only stock-related cached data."""
        st.cache_data.clear()
        st.success("✅ Stock cache data has been cleared")
    
    @staticmethod
    def get_cache_stats() -> dict:
        """
        Get statistics about cached data.
        
        Returns:
            Dictionary with cache statistics
        """
        # Note: Streamlit doesn't provide direct cache stats API
        # This is a placeholder for future implementation
        return {
            'status': 'Cache system active',
            'ttl_data': '1 hour',
            'ttl_sector': '24 hours'
        }
    
    @staticmethod
    def cache_with_key(key: str, ttl: int = 3600):
        """
        Decorator for caching with custom key and TTL.
        
        Args:
            key: Cache key identifier
            ttl: Time to live in seconds
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            @st.cache_data(ttl=ttl)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    @staticmethod
    def invalidate_on_date_change(
        current_start: Any,
        current_end: Any,
        session_key: str = 'last_date_range'
    ) -> bool:
        """
        Check if date range has changed and invalidate cache if needed.
        
        Args:
            current_start: Current start date
            current_end: Current end date
            session_key: Session state key for storing last date range
            
        Returns:
            True if cache was invalidated, False otherwise
        """
        if session_key not in st.session_state:
            st.session_state[session_key] = (current_start, current_end)
            return False
        
        last_start, last_end = st.session_state[session_key]
        
        if last_start != current_start or last_end != current_end:
            st.cache_data.clear()
            st.session_state[session_key] = (current_start, current_end)
            return True
        
        return False
