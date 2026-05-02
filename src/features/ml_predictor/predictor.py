import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from typing import Dict, Tuple, Optional, Any, List

import logging
import threading

from src.core.models import PredictionResult

logger = logging.getLogger(__name__)

class MLPredictor:
    """
    Machine Learning Predictor for forecasting price movements using Regression models.
    """
    
    @staticmethod
    def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate machine learning features from OHLCV data.
        
        Args:
            df: DataFrame containing Open, High, Low, Close, Volume
            
        Returns:
            DataFrame containing engineered features and the 'Target' variable
        """
        if df.empty or len(df) < 100:
            raise ValueError("Insufficient data: At least 100 days of data required for feature engineering.")
            
        features = df.copy()
        
        # Ensure we have required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in features.columns:
                # If only Close is available, mock others
                if 'Close' in features.columns:
                    features[col] = features['Close']
                else:
                    raise ValueError(f"Missing required column: {col}")
        
        # Calculate returns
        features['Return_1d'] = features['Close'].pct_change(1)
        features['Return_5d'] = features['Close'].pct_change(5)
        features['Return_20d'] = features['Close'].pct_change(20)
        
        # Technical Indicators
        # Moving Averages
        features['MA_20'] = features['Close'].rolling(window=20).mean()
        features['MA_50'] = features['Close'].rolling(window=50).mean()
        # Adaptive window: use 200 days if available, otherwise use half the data
        ma_200_window = min(200, max(50, len(features) // 2))
        features['MA_200'] = features['Close'].rolling(window=ma_200_window).mean()
        
        # Volatility
        features['Vol_20d'] = features['Return_1d'].rolling(window=20).std()
        
        # MACD (Approximation using EMA)
        ema_12 = features['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = features['Close'].ewm(span=26, adjust=False).mean()
        features['MACD'] = ema_12 - ema_26
        
        # RSI — Wilder's Smoothing Method (industry standard, EMA with alpha=1/14)
        delta = features['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        features['RSI'] = 100 - (100 / (1 + rs))
        
        # Cyclical Time Features
        if isinstance(features.index, pd.DatetimeIndex):
            day_of_week = features.index.dayofweek
            month = features.index.month
            
            features['Day_Sin'] = np.sin(2 * np.pi * day_of_week / 7)
            features['Day_Cos'] = np.cos(2 * np.pi * day_of_week / 7)
            features['Month_Sin'] = np.sin(2 * np.pi * month / 12)
            features['Month_Cos'] = np.cos(2 * np.pi * month / 12)
        else:
            # Fallback
            features['Day_Sin'] = 0
            features['Day_Cos'] = 0
            features['Month_Sin'] = 0
            features['Month_Cos'] = 0
            
        # Target variable: Next day's closing price
        features['Target'] = features['Close'].shift(-1)
        
        # Drop rows with NaNs (from rolling windows and shifts)
        features = features.dropna()
        
        if len(features) < 10:
            raise ValueError("Insufficient data after dropping NaNs for feature engineering.")
            
        return features

    @staticmethod
    def prepare_sequences(X: np.ndarray, y: np.ndarray, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare 3D sequences for LSTM model.

        Alignment: X_seq[i] contains features for days [i .. i+seq-1].
                   y_seq[i] is the target for day i+seq-1, i.e. the next-day
                   close AFTER the last feature day — one step ahead, not two.

        Fix: use y[i + sequence_length - 1] (not y[i + sequence_length]) so that
        the model predicts exactly one step into the future.
        """
        if len(X) <= sequence_length:
            raise ValueError(f"Not enough data to create sequences of length {sequence_length}")

        X_seq, y_seq = [], []
        for i in range(len(X) - sequence_length):
            X_seq.append(X[i : i + sequence_length])
            y_seq.append(y[i + sequence_length - 1])  # ← fixed: 1-step ahead, not 2

        return np.array(X_seq), np.array(y_seq)
        
    @staticmethod
    def split_data(features_df: pd.DataFrame, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, StandardScaler]:
        """
        Split data strictly temporally (no shuffle) and scale features.
        
        Returns:
            X_train, X_test, y_train, y_test, X_scaler, y_scaler
        """
        X = features_df.drop(columns=['Target']).values
        y = features_df['Target'].values.reshape(-1, 1)
        
        split_idx = int(len(features_df) * (1 - test_size))
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        X_scaler = StandardScaler()
        X_train = X_scaler.fit_transform(X_train)
        X_test = X_scaler.transform(X_test)
        
        y_scaler = StandardScaler()
        y_train = y_scaler.fit_transform(y_train)
        y_test = y_scaler.transform(y_test)
        
        return X_train, X_test, y_train.ravel(), y_test.ravel(), X_scaler, y_scaler

    @staticmethod
    def train_linear_regression(X_train: np.ndarray, y_train: np.ndarray) -> Ridge:
        """Train a Ridge Regression model."""
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        return model

    @staticmethod
    def train_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestRegressor:
        """Train a Random Forest Regressor."""
        model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42)
        model.fit(X_train, y_train)
        return model
        
    @staticmethod
    def train_lstm(X_train: np.ndarray, y_train: np.ndarray, timeout_seconds: int = 60) -> Any:
        """
        Train an LSTM model with Keras.

        Architecture: 2×LSTM(64) + Dropout(0.1) + Dense(1)
        Training:     Adam(0.001) + EarlyStopping(patience=10)
                      + ReduceLROnPlateau for adaptive learning rate
        """
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        except ImportError:
            raise ImportError("TensorFlow is not installed. LSTM cannot be trained.")

        # Ensure 3D shape (samples, timesteps, features)
        if len(X_train.shape) == 2:
            X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))

        model = Sequential([
            LSTM(64, return_sequences=True,
                 input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.1),           # lower dropout — better for smaller datasets
            LSTM(64, return_sequences=False),
            Dropout(0.1),
            Dense(25, activation='relu'),   # intermediate dense layer
            Dense(1)
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse'
        )

        import time
        class TimeoutCallback(tf.keras.callbacks.Callback):
            def __init__(self, timeout):
                super().__init__()
                self.timeout = timeout
                self.start_time = time.time()
            def on_epoch_end(self, epoch, logs=None):
                if time.time() - self.start_time > self.timeout:
                    self.model.stop_training = True
                    logger.warning("LSTM Training timed out.")

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,                # was 5 — less aggressive stopping
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,                 # halve LR when plateau detected
                patience=5,
                min_lr=1e-6,
                verbose=0
            ),
            TimeoutCallback(timeout_seconds),
        ]

        model.fit(
            X_train, y_train,
            epochs=100,                     # more epochs; EarlyStopping decides when to stop
            batch_size=32,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=0
        )

        return model

    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> Dict[str, float]:
        """Calculate evaluation metrics for regression models."""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Handle zero division for MAPE
        mask = y_true != 0
        if np.any(mask):
            mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask])
        else:
            mape = 0.0
            
        return {
            "model_name": model_name,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "r2": r2
        }
        
    @staticmethod
    def predict_horizon(
        model: Any,
        scaler_X: StandardScaler,
        scaler_y: StandardScaler,
        df: pd.DataFrame,
        horizon: int,
        is_lstm: bool = False,
        sequence_length: int = 60,
        residual_std: Optional[float] = None,
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        Make multi-step predictions for the specified horizon using autoregressive forecasting.

        Args:
            residual_std: Standard deviation of model residuals on the test set.
                          When provided, confidence intervals are derived from actual
                          model errors (statistically sound). Falls back to a 2% heuristic
                          if None.

        Returns:
            Tuple of (predicted_prices, lower_bounds, upper_bounds)
        """
        predictions = []

        # ── Optimization: trim DataFrame to a rolling buffer ─────────────────
        # We only need ~200 days of history for MA_200 plus a small buffer.
        # This reduces per-step feature computation from O(n) to O(buffer).
        BUFFER = max(260, sequence_length + 70)
        current_df = df.tail(BUFFER).copy()

        for step in range(horizon):
            features = MLPredictor.prepare_features(current_df)

            if is_lstm:
                last_features = features.drop(columns=['Target']).iloc[-sequence_length:].values
                scaled_X = scaler_X.transform(last_features)
                scaled_X = scaled_X.reshape((1, scaled_X.shape[0], scaled_X.shape[1]))
            else:
                last_features = features.drop(columns=['Target']).iloc[-1:].values
                scaled_X = scaler_X.transform(last_features)

            pred_scaled = model.predict(scaled_X)
            pred_scalar = np.array(pred_scaled).ravel()[0]
            pred = scaler_y.inverse_transform([[pred_scalar]])[0][0]
            predictions.append(float(pred))

            # ── Build the next synthetic row ──────────────────────────────────
            next_date = current_df.index[-1] + pd.Timedelta(days=1)
            if next_date.weekday() >= 5:  # skip weekends
                next_date += pd.Timedelta(days=7 - next_date.weekday())

            new_row = current_df.iloc[-1].copy()
            new_row['Close'] = float(pred)
            new_row['Open']  = float(pred)
            new_row['High']  = float(pred) * 1.01
            new_row['Low']   = float(pred) * 0.99
            # Use rolling-average volume instead of a stale single-day value
            new_row['Volume'] = float(current_df['Volume'].tail(20).mean())

            current_df.loc[next_date] = new_row

            # Keep buffer size manageable: trim the oldest row after each step
            if len(current_df) > BUFFER + horizon:
                current_df = current_df.iloc[1:]

        # ── Confidence Intervals ─────────────────────────────────────────────
        # Use actual model residuals (std of test-set errors) when available;
        # fall back to a conservative 2% heuristic otherwise.
        lower_bounds, upper_bounds = [], []
        for i, p in enumerate(predictions):
            if residual_std is not None and residual_std > 0:
                # Statistically grounded: uncertainty grows with sqrt(horizon)
                uncertainty = residual_std * np.sqrt(i + 1)
            else:
                # Heuristic fallback: 2% base, growing with sqrt(horizon)
                uncertainty = abs(p) * 0.02 * np.sqrt(i + 1)
            lower_bounds.append(p - 1.96 * uncertainty)
            upper_bounds.append(p + 1.96 * uncertainty)

        return predictions, lower_bounds, upper_bounds
