import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import time
from pathlib import Path
import os

# Configuration
TICKER = "GOOG"
START_DATE = "2020-01-01"
END_DATE = "2023-12-31"
SEQ_LENGTH = 60
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "Stock_Prediction_Model.keras"
DATA_CACHE = "data_cache.pkl"

def ensure_dirs():
    """Create required directories"""
    os.makedirs(MODEL_DIR, exist_ok=True)

def robust_download():
    """Download data with retries and caching"""
    # Try to load cached data first
    if os.path.exists(DATA_CACHE):
        data = pd.read_pickle(DATA_CACHE)
        print("Loaded cached data")
        return data
    
    # Download with retries
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"Download attempt {attempt + 1}")
            data = yf.download(
                TICKER,
                start=START_DATE,
                end=END_DATE,
                progress=False,
                timeout=10
            )
            
            if not data.empty:
                data.to_pickle(DATA_CACHE)
                return data
                
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(5)
    
    # Fallback to synthetic data if all downloads fail
    print("Using synthetic data")
    dates = pd.date_range(START_DATE, END_DATE)
    return pd.DataFrame({
        'Close': np.cumsum(np.random.normal(0, 1, len(dates))) + 100
    }, index=dates)

def main():
    ensure_dirs()
    
    # Get data
    data = robust_download()['Close'].values.reshape(-1, 1)
    
    # Proceed with training
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(SEQ_LENGTH, len(scaled)):
        X.append(scaled[i-SEQ_LENGTH:i])
        y.append(scaled[i])
    
    X, y = np.array(X), np.array(y)
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, batch_size=32)
    
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()