from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS  # Add this import
import pandas as pd
import numpy as np
from scipy import stats
import logging
import os
import json
import plotly.graph_objects as go
import plotly.io as pio
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch import arch_model
from scipy.optimize import minimize
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import InterpolationWarning
import warnings
import requests
import math
from datetime import datetime, timedelta
import time
from requests.exceptions import RequestException
import pickle
from pathlib import Path
from functools import wraps

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)  # Add this line to enable CORS for all routes

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Constants for caching
CACHE_FILE = 'btc_data_cache.pkl'
CACHE_EXPIRY = 3600  # Cache expiry time in seconds (1 hour)
PRICE_CACHE_DURATION = 60  # Cache the latest price for 60 seconds
API_CALL_LIMIT = 10  # Maximum number of API calls per minute
API_CALL_INTERVAL = 60  # Time interval for API call limit in seconds

# Global variables
df = None
data_initialized = False
latest_price_cache = {'price': None, 'timestamp': 0}
api_call_count = 0
api_call_reset_time = 0

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if pd.isna(obj):
            return None
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj) if not math.isnan(obj) else None
        return super(NpEncoder, self).default(obj)

def get_data():
    try:
        # Fetch hourly BTC/USD data from Binance API
        symbol = 'BTCUSDT'
        hourly_interval = '1h'
        daily_interval = '1d'
        
        # Function to fetch data for a specific interval
        def fetch_interval_data(interval):
            url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}'
            response = requests.get(url)
            response.raise_for_status()
            data = pd.DataFrame(response.json(),
                                columns=['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time',
                                         'Quote_asset_volume', 'Number_of_trades', 'Taker_buy_base_asset_volume',
                                         'Taker_buy_quote_asset_volume', 'Ignore'])
            data['timestamp'] = pd.to_datetime(data['Open_time'], unit='ms')
            data.set_index('timestamp', inplace=True)
            data = data[['Close']].astype(float)
            data.index.freq = pd.infer_freq(data.index)
            return data

        # Fetch hourly and daily data
        hourly_data = fetch_interval_data(hourly_interval)
        daily_data = fetch_interval_data(daily_interval)
        
        app.logger.info(f"Fetched hourly data shape: {hourly_data.shape}")
        app.logger.info(f"Hourly date range: {hourly_data.index.min()} to {hourly_data.index.max()}")
        app.logger.info(f"Fetched daily data shape: {daily_data.shape}")
        app.logger.info(f"Daily date range: {daily_data.index.min()} to {daily_data.index.max()}")
        
        return {'hourly': hourly_data, 'daily': daily_data}
    except Exception as e:
        app.logger.error(f"Error fetching data from Binance API: {str(e)}")
        raise

def load_cached_data():
    cache_path = Path(CACHE_FILE)
    if cache_path.exists():
        cache_age = time.time() - cache_path.stat().st_mtime
        if cache_age < CACHE_EXPIRY:
            with open(CACHE_FILE, 'rb') as f:
                cached_data = pickle.load(f)
                if isinstance(cached_data, dict) and 'hourly' in cached_data and 'daily' in cached_data:
                    return cached_data
                else:
                    app.logger.warning("Cached data is not in the expected format. Fetching new data.")
    return None

def save_cached_data(data):
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(data, f)

def fetch_and_preprocess_data(max_retries=3, retry_delay=5):
    cached_data = load_cached_data()
    if cached_data is not None:
        app.logger.info("Using cached data")
        return cached_data

    for attempt in range(max_retries):
        try:
            app.logger.info(f"Attempting to fetch data (attempt {attempt + 1}/{max_retries})")
            data = get_data()
            
            if data['hourly'].empty or data['daily'].empty:
                raise ValueError("Fetched data is empty")
            
            for key in ['hourly', 'daily']:
                data[key]['close'] = data[key]['Close']
                data[key]['log_return'] = np.log(data[key]['close'] / data[key]['close'].shift(1))
                data[key].dropna(inplace=True)
            
            app.logger.info(f"Preprocessed hourly data shape: {data['hourly'].shape}")
            app.logger.info(f"Preprocessed daily data shape: {data['daily'].shape}")
            
            save_cached_data(data)
            return data
        except (RequestException, ValueError) as e:
            app.logger.error(f"Error fetching or preprocessing data: {str(e)}")
            if attempt < max_retries - 1:
                app.logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                app.logger.error("Max retries reached. Unable to fetch data.")
                raise

def initialize_data():
    global df
    try:
        df = fetch_and_preprocess_data()
        if df is None or 'hourly' not in df or 'daily' not in df:
            raise ValueError("Data initialization failed: invalid data structure")
        app.logger.info(f"Hourly data initialized successfully. Shape: {df['hourly'].shape}")
        app.logger.info(f"Daily data initialized successfully. Shape: {df['daily'].shape}")
        app.logger.info(f"Hourly columns: {df['hourly'].columns.tolist()}")
        app.logger.info(f"Daily columns: {df['daily'].columns.tolist()}")
        app.logger.info(f"Hourly index range: {df['hourly'].index.min()} to {df['hourly'].index.max()}")
        app.logger.info(f"Daily index range: {df['daily'].index.min()} to {df['daily'].index.max()}")
    except Exception as e:
        app.logger.error(f"Failed to initialize data: {str(e)}")
        df = None

@app.before_request
def before_request():
    global df
    if df is None:
        initialize_data()

def perform_stationarity_analysis(returns):
    adf_result = adfuller(returns)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=InterpolationWarning)
        kpss_result = kpss(returns)
    
    return {
        'adf_statistic': float(adf_result[0]),
        'adf_pvalue': float(adf_result[1]),
        'kpss_statistic': float(kpss_result[0]),
        'kpss_pvalue': float(kpss_result[1]),
        'is_stationary': bool(adf_result[1] < 0.05 and kpss_result[1] > 0.05)  # Convert to Python bool
    }

def perform_autocorrelation_analysis(returns):
    lb_result = acorr_ljungbox(returns, lags=[10], return_df=True)
    acf_values = acf(returns, nlags=20)
    pacf_values = pacf(returns, nlags=20)
    
    return {
        'ljung_box_statistic': float(lb_result['lb_stat'].iloc[0]),
        'ljung_box_pvalue': float(lb_result['lb_pvalue'].iloc[0]),
        'acf': [float(x) for x in acf_values],  # Convert to list of floats
        'pacf': [float(x) for x in pacf_values],  # Convert to list of floats
        'has_significant_autocorrelation': bool(lb_result['lb_pvalue'].iloc[0] < 0.05)  # Convert to Python bool
    }

def perform_distribution_analysis(returns):
    jb_stat, jb_pvalue = stats.jarque_bera(returns)
    
    return {
        'skewness': float(stats.skew(returns)),
        'kurtosis': float(stats.kurtosis(returns)),
        'jarque_bera_statistic': float(jb_stat),
        'jarque_bera_pvalue': float(jb_pvalue),
        'is_normal': bool(jb_pvalue > 0.05),  # Convert to Python bool
        'percentiles': {
            '1%': float(np.percentile(returns, 1)),
            '5%': float(np.percentile(returns, 5)),
            '95%': float(np.percentile(returns, 95)),
            '99%': float(np.percentile(returns, 99))
        }
    }

def calculate_risk_metrics(returns):
    var_95 = float(np.percentile(returns, 5))
    cvar_95 = float(returns[returns <= var_95].mean())
    sharpe_ratio = float((returns.mean() / returns.std()) * np.sqrt(252))  # Assuming daily returns
    return {
        'var_95': var_95,
        'cvar_95': cvar_95,
        'sharpe_ratio': sharpe_ratio
    }

def fit_arima_model(returns):
    model = ARIMA(returns, order=(1,0,1), freq=returns.index.freq)
    results = model.fit()
    forecast = results.forecast(steps=1)
    return {
        'aic': float(results.aic),
        'bic': float(results.bic),
        'forecast': float(forecast.iloc[0])
    }

def replace_nan_with_none(obj):
    if isinstance(obj, dict):
        return {k: replace_nan_with_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_nan_with_none(v) for v in obj]
    elif isinstance(obj, np.generic):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj) if not math.isnan(obj) else None
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    return obj

def calculate_percentile_rank(series, value):
    return stats.percentileofscore(series, value)

def rate_limited(max_per_interval, interval):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            global api_call_count, api_call_reset_time
            now = time.time()
            if now > api_call_reset_time + interval:
                api_call_count = 0
                api_call_reset_time = now
            if api_call_count >= max_per_interval:
                raise Exception("API rate limit exceeded")
            api_call_count += 1
            return func(*args, **kwargs)
        return wrapper
    return decorator

@rate_limited(API_CALL_LIMIT, API_CALL_INTERVAL)
def get_latest_price_from_api():
    url = "https://api.binance.com/api/v3/ticker/price"
    params = {"symbol": "BTCUSDT"}
    response = requests.get(url, params=params)
    data = response.json()
    return float(data['price'])

def get_latest_price():
    global latest_price_cache
    now = time.time()
    if now - latest_price_cache['timestamp'] < PRICE_CACHE_DURATION:
        return latest_price_cache['price']
    
    try:
        price = get_latest_price_from_api()
        latest_price_cache['price'] = price
        latest_price_cache['timestamp'] = now
        return price
    except Exception as e:
        app.logger.warning(f"Failed to get latest price from API: {str(e)}")
        # Use the last price from our dataset
        return df['hourly']['close'].iloc[-1]

def get_todays_return():
    now = datetime.now()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    
    latest_price = get_latest_price()
    
    # Get the closing price from yesterday
    yesterday_close = df['hourly']['close'].loc[:today_start].iloc[-1]
    
    today_return = np.log(latest_price / yesterday_close)
    
    # Add logging
    app.logger.info(f"Today's calculation: latest_price={latest_price}, yesterday_close={yesterday_close}, return={today_return}")
    
    return today_return

def find_return_rank(returns, today_return):
    # Sort returns in descending order of absolute value
    sorted_returns = returns.abs().sort_values(ascending=False)
    
    # Find the rank of today's return
    abs_today_return = abs(today_return)
    if abs_today_return > sorted_returns.max():
        rank = 1
    elif abs_today_return < sorted_returns.min():
        rank = len(sorted_returns)
    else:
        rank = sorted_returns.index.get_loc(sorted_returns.index[sorted_returns >= abs_today_return][0]) + 1
    
    return rank, len(sorted_returns)

def compare_todays_return(returns):
    today_return = get_todays_return()
    
    # Create a 365-day rolling window
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=365)
    
    # Ensure we have data for the full 365-day period
    if start_date < returns.index[0]:
        start_date = returns.index[0]
        app.logger.warning(f"Not enough historical data for a full 365-day window. Using data from {start_date} to {end_date}")
    
    rolling_window = returns.loc[start_date:end_date]
    
    if rolling_window.empty:
        app.logger.warning("Rolling window is empty, using all available returns")
        rolling_window = returns
    
    # Separate positive and negative returns
    positive_returns = rolling_window[rolling_window > 0]
    negative_returns = rolling_window[rolling_window < 0]
    
    if today_return > 0:
        percentile = stats.percentileofscore(positive_returns, today_return)
        comparison_returns = positive_returns
        largest_return = positive_returns.max()
    else:
        percentile = stats.percentileofscore(negative_returns.abs(), abs(today_return))
        comparison_returns = negative_returns
        largest_return = negative_returns.min()  # Largest negative return is the minimum value
    
    rank = (comparison_returns.abs() >= abs(today_return)).sum()
    total_days = len(comparison_returns)
    
    # Compare to largest return
    ratio_to_largest = abs(today_return) / abs(largest_return)
    
    # Add these debug lines
    app.logger.info(f"Rolling window start: {rolling_window.index.min()}, end: {rolling_window.index.max()}")
    app.logger.info(f"Rolling window shape: {rolling_window.shape}")
    app.logger.info(f"Positive returns range: {positive_returns.min()} to {positive_returns.max()}")
    app.logger.info(f"Negative returns range: {negative_returns.min()} to {negative_returns.max()}")
    app.logger.info(f"Today's return: {today_return}")
    app.logger.info(f"Today's return rank: {rank} out of {total_days} {'positive' if today_return > 0 else 'negative'} days")
    app.logger.info(f"Largest {'positive' if today_return > 0 else 'negative'} return: {largest_return}")
    app.logger.info(f"Ratio to largest return: {ratio_to_largest}")
    
    return {
        'today_return': float(today_return),
        'percentile': float(percentile),
        'direction': 'up' if today_return > 0 else 'down',
        'latest_price': get_latest_price(),
        'sanity_check': {
            'min_return': float(rolling_window.min()),
            'max_return': float(rolling_window.max()),
            'median_return': float(rolling_window.median()),
            'positive_returns_10th_percentile': float(positive_returns.quantile(0.1)),
            'positive_returns_90th_percentile': float(positive_returns.quantile(0.9)),
            'negative_returns_10th_percentile': float(negative_returns.quantile(0.1)),
            'negative_returns_90th_percentile': float(negative_returns.quantile(0.9)),
            'rank': int(rank),
            'total_days': int(total_days),
            'is_largest': rank == 1,
            'larger_than_90_percent': rank <= total_days // 10,
            'comparison_type': 'positive' if today_return > 0 else 'negative',
            'largest_return': float(largest_return),
            'ratio_to_largest': float(ratio_to_largest)
        }
    }

def get_return_data():
    try:
        end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = end_date - timedelta(days=365)
        
        hourly_returns = df['hourly']['log_return'].loc[start_date:end_date].dropna()
        daily_returns = df['daily']['log_return'].loc[start_date:end_date].resample('D').sum().dropna()
        
        def rank_returns(returns):
            abs_returns = returns.abs()
            ranks = abs_returns.rank(method='min', ascending=False)
            total_count = len(returns)
            return ranks, total_count

        hourly_ranks, hourly_total = rank_returns(hourly_returns)
        daily_ranks, daily_total = rank_returns(daily_returns)
        
        def prepare_data(returns, ranks, total_count):
            data = [
                {
                    'time': index.strftime('%Y-%m-%d %H:%M' if isinstance(index, pd.Timestamp) else '%Y-%m-%d'),
                    'return': float(ret),
                    'rank': int(ranks[index]),
                    'total': total_count
                }
                for index, ret in returns.items()
            ]
            # Reverse the list to have the latest data on top
            return list(reversed(data))
        
        return {
            'hourly': prepare_data(hourly_returns, hourly_ranks, hourly_total),
            'daily': prepare_data(daily_returns, daily_ranks, daily_total)
        }
    except Exception as e:
        logging.error(f"Error in get_return_data: {str(e)}")
        logging.error(f"hourly_returns shape: {hourly_returns.shape}")
        logging.error(f"daily_returns shape: {daily_returns.shape}")
        raise

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

# Add this new route to serve static files
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

@app.route('/api/data')
def get_api_data():
    app.logger.info("API data route accessed")
    global df
    
    max_retries = 3
    for attempt in range(max_retries):
        if df is None or 'hourly' not in df or 'daily' not in df or df['hourly'].empty or df['daily'].empty:
            app.logger.warning(f"Data not available. Attempting to fetch... (Attempt {attempt + 1}/{max_retries})")
            try:
                initialize_data()
                if df is None or 'hourly' not in df or 'daily' not in df or df['hourly'].empty or df['daily'].empty:
                    raise ValueError("Data is still invalid after initialization")
            except Exception as e:
                app.logger.error(f"Failed to fetch data: {str(e)}")
                if attempt == max_retries - 1:
                    return jsonify({"error": "Unable to fetch data. Please try again later."}), 500
                time.sleep(5)  # Wait 5 seconds before retrying
        else:
            break
    
    try:
        app.logger.info(f"Working with hourly DataFrame of shape: {df['hourly'].shape}")
        app.logger.info(f"Working with daily DataFrame of shape: {df['daily'].shape}")
        
        if df['hourly'].empty or df['daily'].empty:
            raise ValueError("DataFrame is empty")
        
        hourly_returns = df['hourly']['log_return'].dropna()
        daily_returns = df['daily']['log_return'].dropna()
        app.logger.info(f"Number of hourly returns: {len(hourly_returns)}")
        app.logger.info(f"Number of daily returns: {len(daily_returns)}")
        
        if hourly_returns.empty or daily_returns.empty:
            raise ValueError("Returns series is empty")
        
        app.logger.info("Comparing today's return")
        todays_return_comparison = None
        try:
            todays_return_comparison = compare_todays_return(daily_returns)
            app.logger.info(f"Today's return comparison: {todays_return_comparison}")
        except Exception as e:
            app.logger.error(f"Error in compare_todays_return: {str(e)}")
            todays_return_comparison = {"error": str(e)}
        
        app.logger.info("Getting return data")
        return_data = None
        try:
            return_data = get_return_data()
            app.logger.info(f"Return data keys: {return_data.keys() if return_data else 'None'}")
        except Exception as e:
            app.logger.error(f"Error in get_return_data: {str(e)}")
            return_data = {"error": str(e)}
        
        stats_data = {
            'todays_return_comparison': todays_return_comparison,
            'return_data': return_data
        }
        
        app.logger.info("API data successfully generated")
        data_to_return = replace_nan_with_none(stats_data)
        
        # Log the final data structure
        app.logger.info(f"Final data structure: {json.dumps(data_to_return, cls=NpEncoder)}")
        
        return json.dumps(data_to_return, cls=NpEncoder)
    except Exception as e:
        app.logger.error(f"Error generating API data: {str(e)}")
        app.logger.error(f"Hourly DataFrame shape: {df['hourly'].shape if df is not None else 'None'}")
        app.logger.error(f"Daily DataFrame shape: {df['daily'].shape if df is not None else 'None'}")
        app.logger.error(f"Hourly DataFrame columns: {df['hourly'].columns.tolist() if df is not None else 'None'}")
        app.logger.error(f"Daily DataFrame columns: {df['daily'].columns.tolist() if df is not None else 'None'}")
        app.logger.error(f"Hourly DataFrame head: {df['hourly'].head().to_dict() if df is not None else 'None'}")
        app.logger.error(f"Daily DataFrame head: {df['daily'].head().to_dict() if df is not None else 'None'}")
        return jsonify({"error": f"Error generating API data: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)