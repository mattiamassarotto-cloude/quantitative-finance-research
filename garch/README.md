# GARCH Volatility Model

This project implements a GARCH(1,1) model in Python to estimate time-varying volatility in financial returns.

## Main Features

- Downloads historical market data using yfinance
- Computes logarithmic returns
- Estimates a GARCH(1,1) model with Student's t innovations
- Analyses volatility clustering
- Compares the Normal and Student's t distributions
- Identifies the period of maximum estimated volatility

## Technologies

Python, pandas, NumPy, matplotlib, yfinance, arch, SciPy

## Example

The default analysis uses the S&P 500 from 2005 to 2012, including the 2008 financial crisis.
