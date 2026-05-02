import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import tensorflow as tf

from src.features.ml_predictor.predictor import MLPredictor

class TestMLPredictor:
    
    def setup_method(self):
        # Create dummy OHLCV data
        np.random.seed(42)
        days = 250
        dates = pd.date_range(start='2020-01-01', periods=days, freq='B')
        returns = np.random.normal(0, 0.02, days)
        prices = [100.0]
        for r in returns:
            prices.append(prices[-1] * (1 + r))
            
        prices = prices[1:]
        
        self.df = pd.DataFrame({
            'Open': [p * 0.99 for p in prices],
            'High': [p * 1.02 for p in prices],
            'Low': [p * 0.98 for p in prices],
            'Close': prices,
            'Volume': np.random.randint(1000, 1000000, size=days)
        }, index=dates)
        
    def test_prepare_features(self):
        features = MLPredictor.prepare_features(self.df)

        # 1. All expected columns must be present
        expected_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'Return_1d', 'Return_5d', 'Return_20d',
            'MA_20', 'MA_50', 'MA_200',
            'Vol_20d', 'MACD', 'RSI',
            'Day_Sin', 'Day_Cos', 'Month_Sin', 'Month_Cos',
            'Target',
        ]
        for col in expected_cols:
            assert col in features.columns, f"Missing column: {col}"

        # 2. No NaN values after dropna()
        assert features.isna().sum().sum() == 0, "Features still contain NaN values"

        # 3. dropna() must have reduced the number of rows
        assert len(features) < len(self.df), "dropna() should have removed some rows"
        assert len(features) > 0, "Feature engineering produced an empty DataFrame"

        # 4. RSI must be in the valid [0, 100] range
        assert features['RSI'].between(0, 100).all(), "RSI values must be between 0 and 100"

        # 5. Target = next day's Close — verify by checking shift alignment on a subset
        # The last Target value should match the next row's Close in the original df
        # (the very last row is dropped by shift(-1), so we verify the second-to-last)
        original_close = self.df['Close']
        features_close = features['Close']
        features_target = features['Target']
        # For any retained row, Target should equal the next row's Close in features
        assert (features_target.values[:-1] == features_close.values[1:]).all(), \
            "Target must equal the next row's Close (shift -1)"

    def test_prepare_sequences(self):
        X = np.arange(100).reshape(-1, 1)
        y = np.arange(100)
        
        X_seq, y_seq = MLPredictor.prepare_sequences(X, y, sequence_length=10)
        
        assert len(X_seq) == 90
        assert len(y_seq) == 90
        assert X_seq.shape == (90, 10, 1)
        
    def test_split_data(self):
        # We need more data to pass MA_200
        days = 250
        dates = pd.date_range(start='2020-01-01', periods=days, freq='B')
        prices = np.linspace(100, 200, days)
        df = pd.DataFrame({
            'Open': prices * 0.99,
            'High': prices * 1.02,
            'Low': prices * 0.98,
            'Close': prices,
            'Volume': np.random.randint(1000, 1000000, size=days)
        }, index=dates)
        
        features = MLPredictor.prepare_features(df)
        X_train, X_test, y_train, y_test, X_scaler, y_scaler = MLPredictor.split_data(features, test_size=0.2)
        
        assert X_train is not None
        assert y_train is not None
        assert X_scaler is not None
        
    def test_train_models(self):
        X_train = np.random.normal(0, 1, (100, 10))
        y_train = np.random.normal(0, 1, 100)
        
        # Linear Regression
        lr_model = MLPredictor.train_linear_regression(X_train, y_train)
        assert isinstance(lr_model, Ridge)
        
        # Random Forest
        rf_model = MLPredictor.train_random_forest(X_train, y_train)
        assert isinstance(rf_model, RandomForestRegressor)
        
        # LSTM
        # Fast test with small sequence
        X_lstm = X_train.reshape((100, 1, 10))
        lstm_model = MLPredictor.train_lstm(X_lstm, y_train, timeout_seconds=5)
        assert isinstance(lstm_model, tf.keras.models.Sequential)
        
    def test_evaluate(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.0])
        
        metrics = MLPredictor.evaluate(y_true, y_pred, "TestModel")
        
        assert metrics["model_name"] == "TestModel"
        assert "rmse" in metrics
    def test_predict_horizon(self):
        # Setup data
        days = 250
        dates = pd.date_range(start='2020-01-01', periods=days, freq='B')
        prices = np.linspace(100, 200, days)
        df = pd.DataFrame({
            'Open': prices * 0.99,
            'High': prices * 1.02,
            'Low': prices * 0.98,
            'Close': prices,
            'Volume': np.random.randint(1000, 1000000, size=days)
        }, index=dates)
        
        features = MLPredictor.prepare_features(df)
        X_train, X_test, y_train, y_test, X_scaler, y_scaler = MLPredictor.split_data(features, test_size=0.2)
        model = MLPredictor.train_linear_regression(X_train, y_train)
        
        preds, lowers, uppers = MLPredictor.predict_horizon(model, X_scaler, y_scaler, df, horizon=10)
        
        assert len(preds) == 10
        assert len(lowers) == 10
        assert len(uppers) == 10
        assert lowers[0] < preds[0] < uppers[0]
