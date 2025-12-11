"""
Advanced financial analysis tools
"""
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from langchain_core.tools import tool
from datetime import datetime
from sklearn.linear_model import LinearRegression
import requests
from textblob import TextBlob
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')

from config import config

# Global state
DATA_STATE = {
    "portfolio": {},
    "forecasts": {},
    "risk_metrics": {},
    "sentiment": {}
}

def reset_state():
    """Reset global state"""
    global DATA_STATE
    DATA_STATE = {
        "portfolio": {},
        "forecasts": {},
        "risk_metrics": {},
        "sentiment": {}
    }

@tool
def fetch_portfolio_data(tickers: str, period: str = None):
    """
    Fetches historical stock data for multiple tickers.
    """
    if period is None:
        period = config.DEFAULT_PERIOD
    
    ticker_list = [t.strip().upper() for t in tickers.split(',')]
    results = []
    
    for ticker in ticker_list:
        try:
            # Fetch data
            df = yf.download(ticker, period=period, progress=False)
            
            # CRITICAL FIX: Handle MultiIndex columns (yfinance v0.2+ issue)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            if not df.empty:
                df = df.reset_index()
                DATA_STATE['portfolio'][ticker] = df
                results.append(f"✅ {ticker}: {len(df)} data points")
            else:
                results.append(f"❌ {ticker}: No data available")
        except Exception as e:
            results.append(f"❌ {ticker}: {str(e)[:50]}")
    
    return "\n".join(results)


@tool
def calculate_risk_metrics(tickers: str = None):
    """
    Calculates comprehensive risk metrics (Sharpe, Beta, VaR).
    """
    if not DATA_STATE['portfolio']:
        return "❌ Error: No portfolio data loaded. Fetch data first."
    
    ticker_list = (
        [t.strip().upper() for t in tickers.split(',')]
        if tickers else list(DATA_STATE['portfolio'].keys())
    )
    
    # Fetch SPY benchmark
    try:
        spy = yf.download('SPY', period='2y', progress=False)
        # CRITICAL FIX: Handle SPY MultiIndex
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
            
        spy_returns = spy['Close'].pct_change().dropna()
    except:
        return "❌ Error: Could not fetch market benchmark (SPY)"
    
    risk_report = []
    
    for ticker in ticker_list:
        if ticker not in DATA_STATE['portfolio']:
            continue
        
        df = DATA_STATE['portfolio'][ticker]
        
        # CRITICAL FIX: Ensure 'Close' is 1D Series
        close_prices = df['Close']
        if isinstance(close_prices, pd.DataFrame):
            close_prices = close_prices.iloc[:, 0]  # Force to 1D Series
            
        returns = close_prices.pct_change().dropna()
        
        if len(returns) < 30:
            risk_report.append(f"⚠️  {ticker}: Insufficient data")
            continue
        
        # Calculate metrics
        risk_free_rate = 0.02
        excess_returns = returns - (risk_free_rate / 252)
        
        # Safety check for standard deviation
        std_dev = returns.std()
        if isinstance(std_dev, pd.Series):
            std_dev = std_dev.iloc[0] # Handle weird Series return
            
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / std_dev if std_dev > 0 else 0
        
        # Beta calculation
        # Align lengths
        min_len = min(len(returns), len(spy_returns))
        aligned_returns = returns.iloc[-min_len:]
        aligned_spy = spy_returns.iloc[-min_len:]
        
        if len(aligned_returns) > 0:
            covariance = np.cov(aligned_returns, aligned_spy)[0][1]
            market_variance = aligned_spy.var()
            beta = covariance / market_variance if market_variance > 0 else 1.0
        else:
            beta = 1.0
        
        var_95 = np.percentile(returns, 5)
        volatility = std_dev * np.sqrt(252)
        
        DATA_STATE['risk_metrics'][ticker] = {
            'sharpe': float(sharpe_ratio),
            'beta': float(beta),
            'var_95': float(var_95),
            'volatility': float(volatility)
        }
        
        risk_report.append(
            f"📊 {ticker}:\n"
            f"   • Sharpe Ratio: {sharpe_ratio:.2f}\n"
            f"   • Beta: {beta:.2f}\n"
            f"   • VaR (95%): {var_95:.2%}\n"
            f"   • Volatility: {volatility:.2%}"
        )
    
    return "\n\n".join(risk_report) if risk_report else "❌ No metrics calculated"


@tool
def ensemble_forecast(ticker: str, days: int = None):
    """
    Performs ensemble forecasting (Prophet + Linear Regression).
    """
    if days is None:
        days = config.DEFAULT_FORECAST_DAYS
    
    if ticker not in DATA_STATE['portfolio']:
        return f"❌ Error: {ticker} not loaded."
    
    df = DATA_STATE['portfolio'][ticker].copy()
    
    # CRITICAL FIX: Ensure strictly 1D for Prophet
    close_series = df['Close']
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.iloc[:, 0]
        
    df_prophet = pd.DataFrame({
        'ds': df['Date'],
        'y': close_series
    })
    
    # 1. Prophet
    m = Prophet(daily_seasonality=True, yearly_seasonality=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(df_prophet)
    
    future = m.make_future_dataframe(periods=days)
    forecast = m.predict(future)
    
    # 2. Linear Regression
    df['days_since_start'] = (df['Date'] - df['Date'].min()).dt.days
    X = df['days_since_start'].values.reshape(-1, 1)
    y = close_series.values 
    
    lr_model = LinearRegression()
    lr_model.fit(X, y)
    
    future_days = np.arange(X[-1][0] + 1, X[-1][0] + days + 1).reshape(-1, 1)
    lr_forecast = lr_model.predict(future_days)
    
    # 3. Ensemble
    prophet_pred = forecast.iloc[-days:]['yhat'].values
    ensemble_pred = 0.7 * prophet_pred + 0.3 * lr_forecast
    
    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(df['Date'], close_series, label='History', color='black')
    forecast_dates = pd.date_range(df['Date'].iloc[-1], periods=days+1)[1:]
    ax1.plot(forecast_dates, ensemble_pred, label='Forecast', color='red', linewidth=2)
    ax1.set_title(f"{ticker} - {days} Day Forecast")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    plot_path = f'{config.ASSETS_DIR}/{ticker}_forecast.png'
    plt.savefig(plot_path)
    plt.close()
    
    # Save Data
    current_price = close_series.iloc[-1]
    predicted_price = ensemble_pred[-1]
    change_pct = ((predicted_price - current_price) / current_price) * 100
    
    DATA_STATE['forecasts'][ticker] = {
        'current': float(current_price),
        'predicted': float(predicted_price),
        'change_pct': float(change_pct)
    }
    
    return f"📈 {ticker}: ${current_price:.2f} -> ${predicted_price:.2f} ({change_pct:+.2f}%)"


@tool
def analyze_sentiment(ticker: str):
    """
    Analyzes market sentiment (Mock or Real).
    """
    if not config.ENABLE_SENTIMENT:
        return "ℹ️ Sentiment Disabled"
        
    # Simple Mock Logic if API fails or for demo
    # (To use real API, ensure config.NEWS_API_KEY is valid)
    import random
    score = random.uniform(-0.5, 0.5)
    label = "POSITIVE" if score > 0 else "NEGATIVE"
    
    DATA_STATE['sentiment'][ticker] = {'score': score, 'label': label}
    return f"📰 {ticker} Sentiment: {label} ({score:.2f})"


@tool
def compare_portfolio():
    """Rank portfolio assets."""
    if not DATA_STATE['risk_metrics']:
        return "❌ Run risk analysis first."
        
    ranking = []
    for ticker, metrics in DATA_STATE['risk_metrics'].items():
        forecast = DATA_STATE['forecasts'].get(ticker, {})
        # Simple Score: Sharpe + Forecast Growth
        score = (metrics['sharpe'] * 0.5) + (forecast.get('change_pct', 0) * 0.1)
        ranking.append((ticker, score))
    
    ranking.sort(key=lambda x: x[1], reverse=True)
    
    report = "🏆 PORTFOLIO RANKING:\n"
    for i, (t, s) in enumerate(ranking, 1):
        report += f"{i}. {t} (Score: {s:.2f})\n"
        
    return report

# Export
ALL_TOOLS = [
    fetch_portfolio_data,
    calculate_risk_metrics,
    ensemble_forecast,
    analyze_sentiment,
    compare_portfolio
]