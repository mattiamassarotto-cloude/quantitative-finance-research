# Capital Asset Pricing Model (CAPM)

This project implements the Capital Asset Pricing Model (CAPM) in Python to estimate the relationship between an individual stock and the overall market.

## Main Features

- Downloads historical market data using yfinance
- Computes monthly logarithmic returns
- Estimates beta using both the covariance formula and linear regression
- Performs Ordinary Least Squares (OLS) regression
- Visualizes the Security Characteristic Line (SCL)
- Reports alpha, beta, R² and statistical significance

## Technologies

Python, Pandas, NumPy, Matplotlib, Statsmodels, yfinance

## Assets

- JPMorgan Chase (JPM)
- S&P 500 (^GSPC)

## Example

The default analysis estimates the CAPM relationship between JPMorgan and the S&P 500 using monthly returns from 2005 to 2012.
