import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from pathlib import Path
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Configuration
MODEL_PATH = Path("models") / "Stock_Prediction_Model.keras"
SEQ_LENGTH = 60  # Should match what you used in training
VALID_TICKERS = ["AAPL", "GOOG", "MSFT", "AMZN", "TSLA", "RELIANCE.NS", "TCS.NS"]

@st.cache_resource
def load_trained_model():
    """Load the pre-trained model with error handling"""
    try:
        model = load_model(MODEL_PATH)
        scaler = MinMaxScaler()
        return model, scaler
    except Exception as e:
        st.error(f"⚠️ Model loading failed: {str(e)}")
        st.error("Please run train_model.py first to train the model")
        st.stop()

def robust_download(ticker, days):
    """Enhanced download function with all safeguards"""
    suffixes = ['', '.NS', '.BO']  # Try US, India NSE, India BSE formats
    max_days = min(days, 365*2)  # Limit to 2 years max
    
    for suffix in suffixes:
        for attempt in range(3):
            try:
                data = yf.download(
                    ticker + suffix,
                    period=f"{max_days}d",
                    progress=False,
                    timeout=10
                )
                if not data.empty:
                    return data
            except Exception as e:
                print(f"Attempt {attempt} failed for {ticker+suffix}: {str(e)}")
                time.sleep(5 * (attempt + 1))  # Exponential backoff
        
    # Fallback to synthetic data if all fails
    st.warning("Using synthetic data for demonstration")
    dates = pd.date_range(end=datetime.now(), periods=max_days)
    return pd.DataFrame({
        'Close': np.random.normal(100, 10, len(dates)),
        'Open': np.random.normal(100, 10, len(dates)),
        'High': np.random.normal(105, 10, len(dates)),
        'Low': np.random.normal(95, 10, len(dates))
    }, index=dates)

def prepare_input_data(data, scaler):
    """Prepare data for model prediction"""
    try:
        scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
        X = np.array([scaled_data[i-SEQ_LENGTH:i] for i in range(SEQ_LENGTH, len(scaled_data))])
        return X, scaled_data
    except Exception as e:
        st.error(f"Data processing error: {str(e)}")
        return None, None

def main():
    st.set_page_config(page_title="Stock Predictor Pro", layout="wide")
    st.title("📊 Stock Price Predictor Pro")
    st.markdown("Predict stock prices using LSTM neural networks")
    
    # Load model
    model, scaler = load_trained_model()
    
    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        ticker = st.text_input("Stock Symbol", "AAPL").upper()
        days_history = st.slider("Historical Days", 30, 365, 180, 
                               help="Number of days of historical data to use")
        predict_days = st.slider("Forecast Days", 1, 30, 7,
                                help="Number of future days to predict")
        
        st.markdown("### Popular Symbols")
        cols = st.columns(3)
        for i, sym in enumerate(VALID_TICKERS[:6]):
            if cols[i%3].button(sym):
                ticker = sym
    
    # Main prediction logic
    if st.button("Run Prediction", type="primary"):
        with st.spinner(f"Fetching {ticker} data and making predictions..."):
            try:
                # Download data
                data = robust_download(ticker, days_history + SEQ_LENGTH)
                if data is None:
                    return
                
                # Prepare data
                X, scaled_data = prepare_input_data(data, scaler)
                if X is None:
                    return
                
                # Make predictions
                predictions = scaler.inverse_transform(model.predict(X)).flatten()
                
                # Prepare visualization data
                valid_dates = data.index[SEQ_LENGTH:]
                df = pd.DataFrame({
                    'Date': valid_dates,
                    'Actual': data['Close'].values[SEQ_LENGTH:],
                    'Predicted': predictions
                })
                
                # Create interactive plot
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=df['Date'],
                    y=df['Actual'],
                    name='Actual Price',
                    line=dict(color='#1f77b4', width=2),
                    hovertemplate='%{x|%b %d, %Y}<br>Price: $%{y:.2f}<extra></extra>'
                ))
                
                # Model predictions
                fig.add_trace(go.Scatter(
                    x=df['Date'],
                    y=df['Predicted'],
                    name='Model Prediction',
                    line=dict(color='#ff7f0e', width=2, dash='dot'),
                    hovertemplate='%{x|%b %d, %Y}<br>Predicted: $%{y:.2f}<extra></extra>'
                ))
                
                # Future forecast
                last_sequence = scaled_data[-SEQ_LENGTH:]
                future_preds = []
                for _ in range(predict_days):
                    next_pred = model.predict(last_sequence.reshape(1, SEQ_LENGTH, 1))
                    future_preds.append(scaler.inverse_transform(next_pred)[0][0])
                    last_sequence = np.append(last_sequence[1:], next_pred[0])
                
                future_dates = pd.date_range(
                    start=df['Date'].iloc[-1] + timedelta(days=1),
                    periods=predict_days
                )
                
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=future_preds,
                    name=f'{predict_days}-Day Forecast',
                    line=dict(color='#2ca02c', width=2, dash='dot'),
                    hovertemplate='%{x|%b %d, %Y}<br>Forecast: $%{y:.2f}<extra></extra>'
                ))
                
                # Update layout
                fig.update_layout(
                    title=f'{ticker} Stock Price Prediction',
                    xaxis_title='Date',
                    yaxis_title='Price ($)',
                    hovermode='x unified',
                    height=600,
                    template='plotly_white',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                # Display the plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Show key metrics
                current_price = data['Close'].iloc[-1]
                last_pred = predictions[-1]
                forecast = future_preds[-1]
                
                cols = st.columns(3)
                cols[0].metric(
                    "Current Price", 
                    f"${current_price:.2f}",
                    help="Most recent closing price"
                )
                cols[1].metric(
                    "Model Prediction", 
                    f"${last_pred:.2f}",
                    delta=f"{(last_pred/current_price-1)*100:.2f}%",
                    help="Model's prediction for current day"
                )
                cols[2].metric(
                    f"{predict_days}-Day Forecast", 
                    f"${forecast:.2f}",
                    delta=f"{(forecast/current_price-1)*100:.2f}%",
                    help=f"Projected price in {predict_days} days"
                )
                
                # Raw data expander
                with st.expander("📊 View Detailed Data"):
                    st.dataframe(
                        df.tail(10).style.format({
                            'Actual': '{:.2f}',
                            'Predicted': '{:.2f}'
                        }),
                        height=300
                    )
                
            except Exception as e:
                st.error(f"🚨 Prediction failed: {str(e)}")
                st.info("Try a different stock symbol or time range")

if __name__ == "__main__":
    main()
    print("Streamlit app is running. Visit http://localhost:8501 to view it.")