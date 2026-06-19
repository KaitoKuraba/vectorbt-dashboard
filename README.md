# VectorBT Backtesting Dashboard

Full-stack quant backtesting dashboard with Flask + VectorBT backend and React + Plotly.js frontend.

## Strategies
- MA Crossover, RSI, Bollinger Bands, MACD
- CDC ActionZone V3
- RSI Divergence / Convergence
- 16 Candlestick Patterns (pure NumPy)

## Architecture
- **Backend**: Flask + VectorBT (Railway)
- **Frontend**: React 18 + Plotly.js (Vercel)

## Features
- Correlation analysis vs Gold, BTC, SPY, SET, etc.
- Next-bar-open execution (no look-ahead bias)
- Monthly returns heatmap, trade log, parameter optimizer
