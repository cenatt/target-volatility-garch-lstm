# Target Volatility Strategy: GARCH vs LSTM

Comparison of GARCH(1,1) and LSTM neural networks for volatility forecasting in a 12% target volatility framework. The backtest is performed on three US-listed assets (SPY, NVDA, PG) with block bootstrap confidence intervals for both Sharpe ratio and RMSE differences.

## Features
- Synchronised 5-day ahead volatility forecasts (GARCH and LSTM)
- Out-of-sample backtest (train/test split: 80/20) from 2015-01-01 to 2026-06-01
- No leverage (weights bounded between 0 and 1)
- Block bootstrap (20-day blocks, 5000 resamples) for Sharpe and RMSE difference tests
- Equity curve and volatility prediction plots
- Summary table with annualised return, Sharpe ratio, max drawdown, and RMSE

## Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/cenatt/target-volatility-garch-lstm.git
cd target-volatility-garch-lstm
pip install -r requirements.txt
