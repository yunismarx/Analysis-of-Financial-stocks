"""
Validation Service - Input validation and error handling
"""
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Any, Optional


class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


class ValidationService:
    """
    Provides validation methods for user inputs and data integrity.
    Handles validation errors with Arabic error messages.
    """
    
    @staticmethod
    def validate_portfolio_weights(weights: Dict[str, float]) -> None:
        """
        Validate that portfolio weights sum to 1.0 (100%).
        
        Args:
            weights: Dictionary mapping ticker to weight
            
        Raises:
            ValidationError: If weights don't sum to 1.0
        """
        if not weights:
            raise ValidationError("Portfolio must contain at least one stock")
        
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValidationError(
                f"Portfolio weights must sum up to 100%. Current sum: {total*100:.2f}%"
            )
        
        # Check for negative weights
        for ticker, weight in weights.items():
            if weight < 0:
                raise ValidationError(
                    f"Weight for {ticker} cannot be negative: {weight*100:.2f}%"
                )
    
    @staticmethod
    def validate_indicator_period(
        period: int,
        min_val: int,
        max_val: int,
        name: str
    ) -> None:
        """
        Validate that indicator period is within acceptable range.
        
        Args:
            period: Period value to validate
            min_val: Minimum acceptable value
            max_val: Maximum acceptable value
            name: Name of the indicator for error message
            
        Raises:
            ValidationError: If period is outside valid range
        """
        if not isinstance(period, int):
            raise ValidationError(
                f"{name} period must be an integer"
            )
        
        if not (min_val <= period <= max_val):
            raise ValidationError(
                f"{name} period must be between {min_val} and {max_val}. Provided: {period}"
            )
    
    @staticmethod
    def validate_data_sufficiency(
        data: pd.DataFrame,
        min_days: int,
        operation: str
    ) -> None:
        """
        Validate that sufficient data exists for operation.
        
        Args:
            data: DataFrame to validate
            min_days: Minimum number of days required
            operation: Name of the operation
            
        Raises:
            ValidationError: If insufficient data
        """
        if data.empty:
            raise ValidationError(f"No data available for {operation}")
        
        if len(data) < min_days:
            raise ValidationError(
                f"{operation} requires at least {min_days} days of data. "
                f"Available data: {len(data)} days"
            )
    
    @staticmethod
    def validate_tickers(tickers: List[str]) -> None:
        """
        Validate ticker list.
        
        Args:
            tickers: List of ticker symbols
            
        Raises:
            ValidationError: If ticker list is invalid
        """
        if not tickers:
            raise ValidationError("At least one stock must be selected")
        
        # Check for empty strings
        if any(not ticker.strip() for ticker in tickers):
            raise ValidationError("Ticker symbols cannot be empty")
    
    @staticmethod
    def validate_date_range(start_date: Any, end_date: Any) -> None:
        """
        Validate date range.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Raises:
            ValidationError: If date range is invalid
        """
        if start_date >= end_date:
            raise ValidationError(
                "Start date must be before end date"
            )
    
    @staticmethod
    def validate_predictions(predictions: np.ndarray) -> bool:
        """
        Validate that predictions contain valid numbers.
        
        Args:
            predictions: Array of predictions
            
        Returns:
            True if valid, False otherwise
        """
        if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
            st.error("❌ Predictions contain invalid values")
            return False
        return True
    
    @staticmethod
    def validate_portfolio_name(
        name: str,
        existing_names: List[str]
    ) -> None:
        """
        Validate portfolio name is unique and non-empty.
        
        Args:
            name: Portfolio name to validate
            existing_names: List of existing portfolio names
            
        Raises:
            ValidationError: If name is invalid or duplicate
        """
        if not name or not name.strip():
            raise ValidationError("Portfolio name cannot be empty")
        
        if name in existing_names:
            raise ValidationError(
                f"Portfolio '{name}' already exists. Please choose a different name"
            )
    
    @staticmethod
    def validate_confidence_level(confidence: float) -> None:
        """
        Validate confidence level is between 0 and 1.
        
        Args:
            confidence: Confidence level
            
        Raises:
            ValidationError: If confidence level is invalid
        """
        if not (0 < confidence < 1):
            raise ValidationError(
                f"Confidence level must be between 0 and 1. Provided: {confidence}"
            )
    
    @staticmethod
    def safe_validate(
        validation_func: callable,
        *args,
        **kwargs
    ) -> bool:
        """
        Safely execute validation function and display error if it fails.
        
        Args:
            validation_func: Validation function to execute
            *args: Arguments for validation function
            **kwargs: Keyword arguments for validation function
            
        Returns:
            True if validation passed, False if it failed
        """
        try:
            validation_func(*args, **kwargs)
            return True
        except ValidationError as e:
            st.error(f"❌ {str(e)}")
            return False
        except Exception as e:
            st.error(f"❌ Validation Error: {str(e)}")
            return False
