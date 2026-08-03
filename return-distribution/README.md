# Financial Return Distribution Analysis

This project analyses the empirical distribution of financial returns and compares it with theoretical probability distributions.

## Main Features

- Downloads historical market data using yfinance
- Computes logarithmic returns
- Estimates descriptive statistics (mean, volatility, skewness and kurtosis)
- Compares the empirical distribution with the Normal and Student's t distributions
- Visualizes fat tails using logarithmic scaling

## Technologies

Python, Pandas, NumPy, Matplotlib, SciPy, yfinance

## Example

The default analysis uses JPMorgan historical returns from 2005 to 2012 to investigate the limitations of the Gaussian assumption in financial markets.
