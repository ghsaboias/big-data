document.addEventListener('DOMContentLoaded', function() {
    fetch('/api/data')
        .then(response => response.json())
        .then(data => {
            document.getElementById('loading').style.display = 'none';
            document.getElementById('dashboard').style.display = 'block';
            
            Plotly.newPlot('price-chart', JSON.parse(data.price_chart));
            Plotly.newPlot('returns-histogram', JSON.parse(data.returns_histogram));
            Plotly.newPlot('qq-plot', JSON.parse(data.qq_plot));
            Plotly.newPlot('volatility-clustering', JSON.parse(data.volatility_clustering));
            Plotly.newPlot('power-law-plot', JSON.parse(data.power_law_plot));
            Plotly.newPlot('multiscale-plot', JSON.parse(data.multiscale_plot));
            
            displayStats(data.stats);
        })
        .catch(error => {
            document.getElementById('loading').style.display = 'none';
            document.getElementById('error').style.display = 'block';
            document.getElementById('error').textContent = 'Error loading data. Please try again later.';
            console.error('Error:', error);
        });
});

function displayStats(stats) {
    const statsDiv = document.getElementById('stats');
    statsDiv.innerHTML = `
        <h2>Statistical Analysis</h2>
        
        <h3>Stationarity Analysis</h3>
        <p>ADF Test p-value: ${stats.stationarity.adf_pvalue.toFixed(4)} (< 0.05 indicates stationarity)</p>
        <p>KPSS Test p-value: ${stats.stationarity.kpss_pvalue.toFixed(4)} (> 0.05 supports stationarity)</p>
        <p>Series is ${stats.stationarity.is_stationary ? 'stationary' : 'non-stationary'}</p>
        
        <h3>Autocorrelation Analysis</h3>
        <p>Ljung-Box Test p-value: ${stats.autocorrelation.ljung_box_pvalue.toFixed(4)} (< 0.05 indicates significant autocorrelation)</p>
        <p>Series ${stats.autocorrelation.has_significant_autocorrelation ? 'has' : 'does not have'} significant autocorrelation</p>
        
        <h3>Distribution Analysis</h3>
        <p>Skewness: ${stats.distribution.skewness.toFixed(4)} (> 0.5 or < -0.5 indicates significant skew)</p>
        <p>Kurtosis: ${stats.distribution.kurtosis.toFixed(4)} (> 3 indicates fat tails)</p>
        <p>Jarque-Bera Test p-value: ${stats.distribution.jarque_bera_pvalue.toFixed(4)} (< 0.05 indicates non-normal distribution)</p>
        <p>Distribution is ${stats.distribution.is_normal ? 'normal' : 'non-normal'}</p>
        
        <h3>Risk Metrics</h3>
        <p>Value at Risk (95%): ${stats.risk_metrics.var_95.toFixed(4)}</p>
        <p>Conditional Value at Risk (95%): ${stats.risk_metrics.cvar_95.toFixed(4)}</p>
        <p>Sharpe Ratio: ${stats.risk_metrics.sharpe_ratio.toFixed(4)} (> 1 indicates good risk-adjusted returns)</p>
        
        <h3>ARIMA Model</h3>
        <p>AIC: ${stats.arima_model.aic.toFixed(2)}</p>
        <p>BIC: ${stats.arima_model.bic.toFixed(2)}</p>
        <p>Next Step Forecast: ${stats.arima_model.forecast.toFixed(6)}</p>
        
        <h3>Multiscale Analysis</h3>
        <p>See the Multiscale Analysis plot for details on different time scales.</p>
    `;
}