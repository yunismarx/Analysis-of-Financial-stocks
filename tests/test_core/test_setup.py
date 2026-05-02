"""
Test that the project setup is correct
"""
import pytest
from src.core import DataManager, CacheManager, ValidationService, SessionState


class TestProjectSetup:
    """Test basic project setup and imports"""
    
    def test_core_imports(self):
        """Test that all core services can be imported"""
        assert DataManager is not None
        assert CacheManager is not None
        assert ValidationService is not None
        assert SessionState is not None
    
    def test_data_manager_instantiation(self):
        """Test that DataManager can be instantiated"""
        dm = DataManager()
        assert dm is not None
    
    def test_cache_manager_methods(self):
        """Test that CacheManager has expected methods"""
        assert hasattr(CacheManager, 'clear_all_caches')
        assert hasattr(CacheManager, 'clear_data_cache')
        assert hasattr(CacheManager, 'get_cache_stats')
    
    def test_validation_service_methods(self):
        """Test that ValidationService has expected methods"""
        assert hasattr(ValidationService, 'validate_portfolio_weights')
        assert hasattr(ValidationService, 'validate_indicator_period')
        assert hasattr(ValidationService, 'validate_data_sufficiency')
        assert hasattr(ValidationService, 'validate_tickers')
    
    def test_session_state_methods(self):
        """Test that SessionState has expected methods"""
        assert hasattr(SessionState, 'initialize')
        assert hasattr(SessionState, 'get_portfolios')
        assert hasattr(SessionState, 'add_portfolio')
        assert hasattr(SessionState, 'get_indicator_config')
