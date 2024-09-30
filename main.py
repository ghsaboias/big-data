from flask import Flask, jsonify, send_from_directory
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

app = Flask(__name__, static_folder='static', static_url_path='')

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load and preprocess data
try:
    df = pd.read_csv('btcusd_historical_data.csv', parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    df['close'] = pd.to_numeric(df['close'])
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df.dropna(inplace=True)
    app.logger.info("Data loaded and preprocessed successfully")
except Exception as e:
    app.logger.error(f"Error loading or preprocessing data: {str(e)}")
    df = pd.DataFrame()

def create_price_chart():
    fig = go.Figure(data=go.Scatter(x=df.index, y=df['close'], mode='lines'))
    fig.update_layout(title='BTC/USD Price Over Time', xaxis_title='Date', yaxis_title='Price (USD)')
    return pio.to_json(fig)

def create_returns_histogram():
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df['log_return'], name='Log Returns', opacity=0.7))
    fig.add_trace(go.Histogram(x=np.random.normal(loc=df['log_return'].mean(), scale=df['log_return'].std(), size=len(df)), name='Normal Distribution', opacity=0.7))
    fig.update_layout(title='Log Returns Distribution vs Normal Distribution', xaxis_title='Log Return', yaxis_title='Frequency', barmode='overlay')
    return pio.to_json(fig)

def create_qq_plot():
    qq = stats.probplot(df['log_return'].dropna(), dist="norm")
    fig = go.Figure(data=go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers'))
    fig.add_trace(go.Scatter(x=qq[0][0], y=qq[0][0], mode='lines', name='y=x'))
    fig.update_layout(title='Q-Q Plot of Log Returns', xaxis_title='Theoretical Quantiles', yaxis_title='Sample Quantiles')
    return pio.to_json(fig)

def create_volatility_clustering_plot():
    returns = df['log_return'].dropna()
    model = arch_model(returns, vol='GARCH', p=1, q=1)
    results = model.fit(disp='off')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index[1:], y=returns, mode='lines', name='Log Returns'))
    fig.add_trace(go.Scatter(x=df.index[1:], y=results.conditional_volatility, mode='lines', name='GARCH(1,1) Volatility'))
    fig.update_layout(title='Volatility Clustering', xaxis_title='Date', yaxis_title='Log Return / Volatility')
    return pio.to_json(fig)

def power_law_fit(data, xmin):
    data = data[data >= xmin]
    def power_law(x, alpha):
        return (alpha - 1) / xmin * (x / xmin) ** -alpha
    def neg_log_likelihood(alpha):
        return -np.sum(np.log(power_law(data, alpha)))
    result = minimize(neg_log_likelihood, 1.5, method='Nelder-Mead')
    return result.x[0]

def create_power_law_plot():
    returns = df['log_return'].abs()
    xmin = returns.quantile(0.95)
    alpha = power_law_fit(returns, xmin)
    
    x = np.logspace(np.log10(xmin), np.log10(returns.max()), 100)
    y = (alpha - 1) / xmin * (x / xmin) ** -alpha
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=returns, y=np.arange(len(returns))[::-1]/len(returns),
                             mode='markers', name='Empirical CCDF'))
    fig.add_trace(go.Scatter(x=x, y=(x/xmin)**(1-alpha), mode='lines', name='Power Law Fit'))
    fig.update_layout(title=f'Power Law Fit (α = {alpha:.2f})',
                      xaxis_title='Log Return (absolute)', yaxis_title='CCDF',
                      xaxis_type="log", yaxis_type="log")
    return pio.to_json(fig)

def multiscale_analysis():
    time_scales = ['1D', '3D', '1W', '2W', '1ME']  # Changed '1M' to '1ME'
    results = {}
    for scale in time_scales:
        resampled = df['close'].resample(scale).last()
        returns = np.log(resampled / resampled.shift(1)).dropna()
        results[scale] = {
            'mean': float(returns.mean()),
            'std': float(returns.std()),
            'skew': float(returns.skew()),
            'kurtosis': float(returns.kurtosis()),
            'jarque_bera_pvalue': float(stats.jarque_bera(returns)[1])
        }
    return results

def create_multiscale_plot():
    analysis = multiscale_analysis()
    scales = list(analysis.keys())
    
    fig = go.Figure()
    for metric in ['std', 'skew', 'kurtosis']:
        fig.add_trace(go.Scatter(x=scales, y=[analysis[scale][metric] for scale in scales],
                                 mode='lines+markers', name=metric.capitalize()))
    
    fig.update_layout(title='Multiscale Analysis',
                      xaxis_title='Time Scale', yaxis_title='Metric Value')
    return pio.to_json(fig)

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
    sharpe_ratio = float((returns.mean() / returns.std()) * np.sqrt(252))
    return {
        'var_95': var_95,
        'cvar_95': cvar_95,
        'sharpe_ratio': sharpe_ratio
    }

def fit_arima_model(returns):
    model = ARIMA(returns, order=(1,0,1))
    results = model.fit()
    forecast = results.forecast(steps=1)
    return {
        'aic': float(results.aic),
        'bic': float(results.bic),
        'forecast': float(forecast.iloc[0])
    }

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/data')
def get_data():
    if df.empty:
        return jsonify({"error": "Data not available"}), 500
    
    returns = df['log_return'].dropna()
    
    stationarity_results = perform_stationarity_analysis(returns)
    autocorrelation_results = perform_autocorrelation_analysis(returns)
    distribution_results = perform_distribution_analysis(returns)
    risk_metrics = calculate_risk_metrics(returns)
    arima_results = fit_arima_model(returns)
    
    stats_data = {
        'stationarity': stationarity_results,
        'autocorrelation': autocorrelation_results,
        'distribution': distribution_results,
        'risk_metrics': risk_metrics,
        'arima_model': arima_results,
        'multiscale_analysis': multiscale_analysis()
    }
    
    return jsonify({
        'price_chart': create_price_chart(),
        'returns_histogram': create_returns_histogram(),
        'qq_plot': create_qq_plot(),
        'volatility_clustering': create_volatility_clustering_plot(),
        'power_law_plot': create_power_law_plot(),
        'multiscale_plot': create_multiscale_plot(),
        'stats': stats_data
    })

if __name__ == '__main__':
    app.run(debug=True)