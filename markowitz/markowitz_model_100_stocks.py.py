import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optimization

stocks = [
    'AAPL','MSFT','NVDA','AMZN','GOOGL','GOOG','JPM','V','MA','XOM',
    'UNH','COST','PG','JNJ','HD','BAC','KO','CVX','WMT','MRK',
    'PEP','NFLX','AMD','ADBE','CRM','CSCO','TMO','ACN','MCD','DHR',
    'ABT','QCOM','TXN','AMGN','INTC','CMCSA','INTU','BKNG','AMAT','MU',
    'ADI','LRCX','VRTX','SBUX','GILD','MDLZ','ADP','KLAC','SNPS','CDNS',
    'MAR','CTAS','REGN','ORLY','CSX','PAYX','MNST','KDP','ROST','AEP',
    'EA','IBM','GE','DIS','BA','CAT','MMM','HON','UPS','LOW',
    'GS','MS','C','AXP','T','VZ','NKE','LMT','DE','USB',
    'BK','BLK','FDX','PFE','BMY','TGT','MO','CL','SO','DUK',
    'EMR','ITW','NSC','COP','SLB','EOG','OXY','AIG','MET','PRU'
]

start_data = '2007-01-01'
end_data = '2009-12-31'
NUM_TRADING_DAYS = 252
num_portfolio = 50000

irx = yf.download('^IRX', start=start_data, end=end_data, auto_adjust=True, progress=False)['Close']
RISK_FREE = irx.mean().item() / 100
print('Risk free =', RISK_FREE)


def download_data():
    frames = []

    for stock in stocks:
        try:
            data = yf.download(stock, start=start_data, end=end_data,
                            auto_adjust=True, progress=False)

            if data.empty:
                print(f"{stock} escluso: dati vuoti")
                continue

            close = data["Close"].dropna()

            if len(close) > 500:
                close.name = stock
                frames.append(close)
            else:
                print(f"{stock} escluso: pochi dati")

        except Exception as e:
            print(f"{stock} escluso: {e}")

    data = pd.concat(frames, axis=1)
    data = data.ffill().dropna()

    print("Numero titoli utilizzati:", data.shape[1])
    print("Numero osservazioni:", data.shape[0])
    print("Ticker utilizzati:", list(data.columns))

    return data


def calculate_return(data):
    returns = np.log(data / data.shift(1)).dropna()
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    return returns


def statistics(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * NUM_TRADING_DAYS
    portfolio_volatility = np.sqrt(
        np.dot(weights.T, np.dot(returns.cov() * NUM_TRADING_DAYS, weights))
    )
    sharpe_ratio = (portfolio_return - RISK_FREE) / portfolio_volatility
    return np.array([portfolio_return, portfolio_volatility, sharpe_ratio])


def generate_portfolio(returns):
    n_assets = len(returns.columns)

    portfolio_weights = []
    portfolio_means = []
    portfolio_risks = []

    for _ in range(num_portfolio):
        # alpha < 1 genera pesi più concentrati, utile con molti asset
        w = np.random.dirichlet(np.ones(n_assets) * 0.25)

        portfolio_weights.append(w)
        stats = statistics(w, returns)

        portfolio_means.append(stats[0])
        portfolio_risks.append(stats[1])

    return (
        np.array(portfolio_weights),
        np.array(portfolio_means),
        np.array(portfolio_risks)
    )


def min_function_sharpe(weights, returns):
    return -statistics(weights, returns)[2]


def optimize_portfolio(returns):
    n_assets = len(returns.columns)
    bounds = tuple((0, 1) for _ in range(n_assets))
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

    best_result = None

    # più punti di partenza per evitare soluzioni instabili
    starting_points = [
        np.repeat(1 / n_assets, n_assets),
        *[np.random.dirichlet(np.ones(n_assets) * 0.25) for _ in range(20)]
    ]

    for x0 in starting_points:
        result = optimization.minimize(
            fun=min_function_sharpe,
            x0=x0,
            args=(returns,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if result.success:
            if best_result is None or result.fun < best_result.fun:
                best_result = result

    return best_result


def optimize_min_variance(returns):
    n_assets = len(returns.columns)
    bounds = tuple((0, 1) for _ in range(n_assets))
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

    x0 = np.repeat(1 / n_assets, n_assets)

    return optimization.minimize(
        fun=lambda w: statistics(w, returns)[1],
        x0=x0,
        args=(),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )


def print_top_weights(optimum, returns, title):
    weights = pd.Series(optimum.x, index=returns.columns)
    weights = weights[weights > 0.001].sort_values(ascending=False)

    print("\n" + title)
    print(weights.head(10).round(4))
    print("Expected return, volatility and Sharpe ratio:",
        statistics(optimum.x, returns).round(4))


def show_optimal_portfolio(max_sharpe, min_var, rets, portfolio_rets, portfolio_vols):
    plt.figure(figsize=(10, 6))

    sharpe = (portfolio_rets - RISK_FREE) / portfolio_vols

    plt.scatter(
        portfolio_vols,
        portfolio_rets,
        c=sharpe,
        cmap='viridis',
        marker='o',
        alpha=0.7,
        s=18
    )

    plt.colorbar(label='Sharpe Ratio')

    plt.grid(True)
    plt.xlabel('Expected volatility')
    plt.ylabel('Expected return')
    plt.title('Monte Carlo simulation of feasible portfolios')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    dataset = download_data()

    if dataset.empty:
        raise ValueError("Dataset vuoto.")

    log_daily_returns = calculate_return(dataset)

    pweights, means, risks = generate_portfolio(log_daily_returns)

    max_sharpe = optimize_portfolio(log_daily_returns)
    min_var = optimize_min_variance(log_daily_returns)

    print_top_weights(max_sharpe, log_daily_returns, "Max Sharpe Portfolio - Top weights")
    print_top_weights(min_var, log_daily_returns, "Minimum Variance Portfolio - Top weights")

    print("\nMonte Carlo range:")
    print("Return min/max:", means.min().round(4), means.max().round(4))
    print("Vol min/max:", risks.min().round(4), risks.max().round(4))

    print("\nOptimized portfolios:")
    print("Max Sharpe:", statistics(max_sharpe.x, log_daily_returns).round(4))
    print("Min Variance:", statistics(min_var.x, log_daily_returns).round(4))

    show_optimal_portfolio(max_sharpe, min_var, log_daily_returns, means, risks)

    print(means.min(), means.max())
    print(risks.min(), risks.max())

    print(statistics(max_sharpe.x, log_daily_returns))
    print(statistics(min_var.x, log_daily_returns))