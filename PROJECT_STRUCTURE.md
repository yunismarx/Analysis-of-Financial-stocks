# Advanced Portfolio Analytics - Project Structure

## Directory Structure

```
.
├── app.py                          # Main Streamlit application (existing)
├── requirements.txt                # Python dependencies
├── pytest.ini                      # Pytest configuration
├── PROJECT_STRUCTURE.md           # This file
│
├── src/                           # Source code
│   ├── __init__.py
│   │
│   ├── core/                      # Core services
│   │   ├── __init__.py
│   │   ├── data_manager.py        # Data fetching and caching
│   │   ├── cache_manager.py       # Cache management utilities
│   │   ├── validation_service.py  # Input validation
│   │   └── session_state.py       # Streamlit session state management
│   │
│   └── features/                  # Feature modules
│       ├── __init__.py
│       │
│       ├── technical_indicators/  # Technical analysis indicators
│       │   └── __init__.py
│       │
│       ├── portfolio_comparator/  # Portfolio comparison
│       │   └── __init__.py
│       │
│       ├── sector_analyzer/       # Sector analysis
│       │   └── __init__.py
│       │
│       └── ml_predictor/          # Machine learning predictions
│           └── __init__.py
│
└── tests/                         # Test suite
    ├── __init__.py
    │
    ├── test_core/                 # Tests for core services
    │   └── __init__.py
    │
    ├── test_features/             # Tests for feature modules
    │   └── __init__.py
    │
    └── property_tests/            # Property-based tests
        └── __init__.py
```

## Module Descriptions

### Core Services (`src/core/`)

**DataManager** (`data_manager.py`)
- Centralized data fetching from Yahoo Finance
- Caching with 1-hour TTL for price data
- Error handling for failed ticker downloads
- Data validation and sufficiency checks
- Sector information retrieval

**CacheManager** (`cache_manager.py`)
- Cache invalidation utilities
- Cache statistics and monitoring
- Custom caching decorators
- Date range change detection

**ValidationService** (`validation_service.py`)
- Portfolio weight validation
- Indicator parameter validation
- Data sufficiency validation
- Ticker and date range validation
- Arabic error messages

**SessionState** (`session_state.py`)
- Streamlit session state management
- Portfolio storage and retrieval
- Indicator configuration persistence
- ML model caching
- Sector classification cache

### Feature Modules (`src/features/`)

**Technical Indicators** (`technical_indicators/`)
- RSI (Relative Strength Index) calculation
- MACD (Moving Average Convergence Divergence)
- Moving Averages (SMA)
- Interactive visualizations with Plotly
- Configurable parameters

**Portfolio Comparator** (`portfolio_comparator/`)
- Multiple portfolio creation (up to 5)
- Performance metrics (return, volatility, Sharpe ratio)
- Risk metrics (VaR, CVaR, max drawdown, beta)
- Comparison visualizations
- Portfolio management (create, edit, delete)

**Sector Analyzer** (`sector_analyzer/`)
- GICS sector classification
- Sector allocation analysis
- Sector performance metrics
- Correlation matrix
- Distribution visualizations

**ML Predictor** (`ml_predictor/`)
- Linear Regression model
- Random Forest model
- LSTM Neural Network
- Feature engineering
- Prediction generation with confidence intervals
- Model comparison and evaluation

## Testing Structure

### Unit Tests (`tests/test_core/`, `tests/test_features/`)
- Test specific functionality with known inputs/outputs
- Edge case testing
- Error handling validation
- Integration testing

### Property-Based Tests (`tests/property_tests/`)
- Universal correctness properties
- Hypothesis framework (minimum 100 iterations)
- Mathematical invariants
- Comprehensive input coverage

## Dependencies

### Core
- streamlit: Web application framework
- pandas: Data manipulation
- numpy: Numerical computing
- yfinance: Stock data fetching

### Visualization
- plotly: Interactive charts

### Statistics
- scipy: Statistical functions

### Machine Learning
- scikit-learn: Traditional ML models
- tensorflow: Deep learning (LSTM)

### Testing
- pytest: Test framework
- hypothesis: Property-based testing
- pytest-cov: Coverage reporting

## Usage

### Running the Application
```bash
streamlit run app.py
```

### Running Tests
```bash
# Run all tests
pytest

# Run unit tests only
pytest -m unit

# Run property tests only
pytest -m property

# Run with coverage
pytest --cov=src --cov-report=html
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```

## Next Steps

1. Implement Technical Indicators module (Task 2-6)
2. Implement Portfolio Comparator module (Task 7-10)
3. Implement Sector Analyzer module (Task 11-13)
4. Implement ML Predictor module (Task 14-19)
5. Integrate all modules into main UI (Task 20)
6. Complete testing and validation (Task 21-22)
