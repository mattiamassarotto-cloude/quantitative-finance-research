# Dynamic Conditional Correlation (DCC-GARCH)

This project implements a Dynamic Conditional Correlation (DCC-GARCH) model in Python to estimate time-varying correlations between financial assets.

## Main Features

- Estimates univariate GARCH(1,1) models for multiple assets
- Computes standardized residuals
- Estimates dynamic conditional correlations using the DCC-GARCH framework
- Analyses correlation behaviour during periods of financial stress
- Evaluates changes in portfolio diversification over time

## Technologies

Python, Pandas, NumPy, Matplotlib, SciPy, ARCH, yfinance

## Assets

- Apple (AAPL)
- JPMorgan Chase (JPM)
- Exxon Mobil (XOM)
- Microsoft (MSFT)
- Johnson & Johnson (JNJ)

## Example

The default analysis estimates dynamic correlations between major U.S. stocks from 2000 to 2012, highlighting the impact of the 2008 financial crisis.
