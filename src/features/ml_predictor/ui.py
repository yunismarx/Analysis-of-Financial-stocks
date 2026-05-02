import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Any

from src.core.models import Portfolio
from src.core.data_manager import DataManager
from src.features.ml_predictor.predictor import MLPredictor

def plot_prediction(historical_prices: pd.Series, predicted_prices: List[float], 
                    lower_bounds: List[float], upper_bounds: List[float], 
                    ticker: str, model_name: str) -> go.Figure:
    """
    Plot historical prices and the predicted future prices with confidence intervals.
    """
    fig = go.Figure()
    
    # Plot historical
    fig.add_trace(go.Scatter(
        x=historical_prices.index,
        y=historical_prices.values,
        mode='lines',
        name='Historical Price',
        line=dict(color='#2196F3', width=2)
    ))
    
    # Generate future dates (business days)
    future_dates = []
    current_date = historical_prices.index[-1]
    for _ in range(len(predicted_prices)):
        current_date += pd.Timedelta(days=1)
        while current_date.weekday() >= 5: # Skip weekends
            current_date += pd.Timedelta(days=1)
        future_dates.append(current_date)
        
    # Plot Confidence Intervals (Shaded Area)
    fig.add_trace(go.Scatter(
        x=future_dates + future_dates[::-1],
        y=upper_bounds + lower_bounds[::-1],
        fill='toself',
        fillcolor='rgba(233, 30, 99, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        name='95% Confidence Interval'
    ))
    
    # Connect last historical to first prediction
    connect_x = [historical_prices.index[-1], future_dates[0]]
    connect_y = [historical_prices.values[-1], predicted_prices[0]]
    
    fig.add_trace(go.Scatter(
        x=connect_x,
        y=connect_y,
        mode='lines',
        line=dict(color='#E91E63', width=2, dash='dash'),
        showlegend=False,
        hoverinfo="skip"
    ))
    
    # Plot predicted path
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=predicted_prices,
        mode='lines+markers',
        name=f'Predicted ({model_name})',
        line=dict(color='#E91E63', width=2),
        marker=dict(size=6, symbol='circle')
    ))
    
    fig.update_layout(
        title=f"{ticker} Price Forecast ({len(predicted_prices)} Days) - {model_name}",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def render_ml_predictor_ui(portfolio: Portfolio, start_date: str, end_date: str):
    """
    Render the UI for Machine Learning Price Predictor.
    """
    st.write("### 🤖 Machine Learning Predictor")
    st.info("💡 **What is this section?** This system studies the stock's past movements using artificial intelligence to try and guess its future price. You can use it to understand the general trend of the stock.")
    
    tickers = portfolio.get_tickers()
    if not tickers:
        st.warning("⚠️ Portfolio is empty. Cannot perform prediction.")
        return
        
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_ticker = st.selectbox("Select Ticker:", tickers)
    with col2:
        model_choice = st.selectbox(
            "Select AI Method:", 
            [
                "Compare All Models (Best for certainty)", 
                "Linear Regression (Ridge) - Simple & Fast", 
                "Random Forest - Accurate for complex patterns", 
                "LSTM (Deep Learning) - Advanced for time series"
            ],
            help="If you are unsure, select 'Compare All Models' and we will try them all and pick the best one for you."
        )
    with col3:
        horizon = st.slider("Prediction Horizon (days):", min_value=7, max_value=90, value=30, step=1)
        
    if st.button("🚀 Train Model and Start Prediction", use_container_width=True):
        
        # 1. Fetch Full Data
        with st.spinner(f"Fetching historical data for {selected_ticker}..."):
            df = DataManager.get_full_ticker_data(selected_ticker, start_date, end_date)
            
        if df.empty or len(df) < 100:
            st.error("❌ Insufficient data for training (less than 100 days).")
            return
            
        # 2. Feature Engineering
        with st.spinner("Performing Feature Engineering..."):
            try:
                features = MLPredictor.prepare_features(df)
            except Exception as e:
                st.error(f"❌ Error during feature engineering: {str(e)}")
                return
                
        # 3. Train-Test Split
        X_train, X_test, y_train, y_test, X_scaler, y_scaler = MLPredictor.split_data(features, test_size=0.2)
        
        is_lstm_only = model_choice == "LSTM (Deep Learning) - Advanced for time series"
        compare_mode = model_choice == "Compare All Models (Best for certainty)"
        sequence_length = 60
        
        # Sequence preparation for LSTM
        X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm = X_train, y_train, X_test, y_test
        
        if is_lstm_only or compare_mode:
            with st.spinner("Preparing sequences for LSTM model..."):
                try:
                    if len(X_train) > sequence_length:
                        X_train_lstm, y_train_lstm = MLPredictor.prepare_sequences(X_train, y_train, sequence_length)
                        X_test_lstm, y_test_lstm = MLPredictor.prepare_sequences(X_test, y_test, sequence_length)
                    elif is_lstm_only:
                        st.error("❌ Insufficient data to train LSTM model (requires longer periods).")
                        return
                    else:
                        # Fallback to 1-day sequence if not enough data in compare mode
                        X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
                        X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
                        sequence_length = 1
                except Exception as e:
                    if is_lstm_only:
                        st.error(f"❌ Sequence error: {str(e)}")
                        return
                    else:
                        sequence_length = 1
        
        # 4. Model Training & Evaluation
        compare_mode = model_choice == "Compare All Models (Best for certainty)"
        models_to_train = ["Linear Regression (Ridge) - Simple & Fast", "Random Forest - Accurate for complex patterns", "LSTM (Deep Learning) - Advanced for time series"] if compare_mode else [model_choice]
        
        trained_models = {}
        model_metrics = {}
        model_residuals = {}  # stores residual std per model for CI calculation
        
        for m_name in models_to_train:
            with st.spinner(f"Training and evaluating model {m_name}..."):
                is_lstm_model = "LSTM" in m_name
                try:
                    # Train
                    if "Linear Regression" in m_name:
                        m = MLPredictor.train_linear_regression(X_train, y_train)
                    elif "Random Forest" in m_name:
                        m = MLPredictor.train_random_forest(X_train, y_train)
                    else: # LSTM
                        st.toast("⚠️ LSTM model needs a few extra seconds for deep training...", icon="⏳")
                        m = MLPredictor.train_lstm(X_train_lstm, y_train_lstm, timeout_seconds=60)
                        
                    trained_models[m_name] = m
                    
                    # Evaluate
                    if is_lstm_model:
                        y_pred_scaled = m.predict(X_test_lstm).ravel()
                        y_true_eval = y_scaler.inverse_transform(y_test_lstm.reshape(-1, 1)).ravel()
                    else:
                        y_pred_scaled = m.predict(X_test)
                        y_true_eval = y_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

                    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

                    metrics = MLPredictor.evaluate(y_true_eval, y_pred, m_name)
                    model_metrics[m_name] = metrics
                    # Residual std in original price units — used for CI calculation
                    model_residuals[m_name] = float(np.std(y_true_eval - y_pred))
                except Exception as e:
                    st.error(f"❌ Error with model {m_name}: {str(e)}")
                    return
                    
        # Select best model if comparing
        if compare_mode:
            best_model_name = min(model_metrics.keys(), key=lambda k: model_metrics[k]['rmse'])
            st.success(f"🏆 Based on tests, the best model for this stock is: **{best_model_name.split(' -')[0]}**")
            final_model = trained_models[best_model_name]
            final_model_name = best_model_name
            final_metrics = model_metrics[best_model_name]
            final_residual_std = model_residuals[best_model_name]
            is_lstm = "LSTM" in final_model_name
        else:
            final_model = trained_models[model_choice]
            final_model_name = model_choice
            final_metrics = model_metrics[model_choice]
            final_residual_std = model_residuals[model_choice]
            is_lstm = "LSTM" in final_model_name
                
        # 6. Prediction for horizon
        display_name = final_model_name.split(' -')[0]
        with st.spinner(f"Calculating predictions for {horizon} days using {display_name}..."):
            try:
                preds, lowers, uppers = MLPredictor.predict_horizon(
                    model=final_model,
                    scaler_X=X_scaler,
                    scaler_y=y_scaler,
                    df=df,
                    horizon=horizon,
                    is_lstm=is_lstm,
                    sequence_length=sequence_length,
                    residual_std=final_residual_std,
                )
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
                return
                
        # --- Visualization & Results ---
        st.success("✅ Training and prediction completed successfully!")
        
        st.subheader("📊 Prediction Results")
        
        last_actual_price = df['Close'].iloc[-1]
        final_predicted_price = preds[-1]
        price_diff = final_predicted_price - last_actual_price
        pct_diff = (price_diff / last_actual_price) * 100
        
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Last Actual Price", f"${last_actual_price:.2f}")
        col_m2.metric(
            f"Predicted Price after {horizon} days", 
            f"${final_predicted_price:.2f}",
            f"{pct_diff:.2f}%",
            delta_color="normal"
        )
        col_m3.metric("Model Accuracy (R²)", f"{final_metrics['r2']*100:.1f}%")
        
        # Display warning if MAPE > 10%
        if final_metrics['mape'] > 0.10:
            st.warning(f"⚠️ The model has relatively low accuracy. The Mean Absolute Percentage Error (MAPE) is {final_metrics['mape']*100:.1f}%, which is above 10%.")
        
        # Plot
        # Show last 100 days for clarity
        plot_df = df['Close'].tail(100)
        fig = plot_prediction(plot_df, preds, lowers, uppers, selected_ticker, final_model_name)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed Metrics
        with st.expander("📈 Want to know how we calculated the model's accuracy? (Click here)"):
            st.markdown("""
            **Terms Guide:**
            - **RMSE (Financial Error):** The smaller this number, the closer the prediction is to the actual stock price in dollars.
            - **MAPE (Percentage Error):** The average deviation of the model's predictions from the real price (e.g. 2% means the model usually misses by 2%).
            - **R² (Model Power):** Measured from 0 to 100%. The closer to 100%, the better the model understands the stock's behavior.
            """)
            if compare_mode:
                comparison_data = []
                for m_name, mets in model_metrics.items():
                    comparison_data.append({
                        "Model Name": m_name.split(' -')[0],
                        "Financial Error (RMSE)": mets['rmse'],
                        "Percentage Error (MAPE) %": mets['mape'] * 100,
                        "Model Power (R²) %": mets['r2'] * 100
                    })
                comp_df = pd.DataFrame(comparison_data).set_index("Model Name")
                
                # Highlight best model row
                def highlight_best(s):
                    is_best = s.name == best_model_name.split(' -')[0]
                    return ['background-color: rgba(0, 120, 212, 0.2); font-weight: bold;' if is_best else '' for _ in s]
                
                st.dataframe(comp_df.style.apply(highlight_best, axis=1).format("{:.2f}"))
                
                st.download_button(
                    "💾 Download Comparison Table (CSV)",
                    data=comp_df.to_csv().encode('utf-8-sig'),
                    file_name="model_comparison.csv",
                    mime="text/csv"
                )
            else:
                metrics_df = pd.DataFrame([
                    {"Metric": "Financial Error (RMSE) - in Dollars", "Value": f"${final_metrics['rmse']:.2f}"},
                    {"Metric": "Percentage Error (MAPE)", "Value": f"{final_metrics['mape']*100:.2f}%"},
                    {"Metric": "Model Power (R²)", "Value": f"{final_metrics['r2']*100:.2f}%"},
                ])
                st.table(metrics_df)
                
        # Export Prediction Results
        pred_df = pd.DataFrame({
            "Day": range(1, len(preds) + 1),
            "Predicted Price": preds,
            "Lower Bound (95%)": lowers,
            "Upper Bound (95%)": uppers
        })
        st.download_button(
            "📥 Download Prediction Data (CSV)",
            data=pred_df.to_csv(index=False).encode('utf-8-sig'),
            file_name=f"{selected_ticker}_forecast.csv",
            mime="text/csv"
        )
