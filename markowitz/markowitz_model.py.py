import numpy as np
import yfinance as yf 
import pandas as pd 
import matplotlib.pyplot as plt
import scipy.optimize as optimization

#stocks we are going to handle
stocks = ['AAPL', 'JPM', 'XOM', 'JNJ', 'MSFT',]

#historical data - define the START and END dates
start_data = '2007-01-01'
end_data = '2009-12-31'
NUM_TRADING_DAYS = 252 
num_portfolio = 10000

irx =  yf.download('^IRX', start = start_data, end= end_data, auto_adjust= True) ['Close']
RISK_FREE = irx.mean().item()/100
print('risk free = ', RISK_FREE)

def download_data():
    #name of the stocks (key)- stock values (2000/2007) as the values    stock_data = {}
    stock_data= {}
    
    for stock in stocks: 
        ticker = yf.Ticker(stock)
        stock_data[stock] = ticker.history(start= start_data, end = end_data, auto_adjust = True)['Close']

    return pd.DataFrame(stock_data).dropna()

def show_data(data):
    normalized_data = data / data.iloc[0] * 100
    normalized_data.plot(figsize = (10,5))
    plt.title('Normalized stock prices (Base = 100)')
    plt.xlabel('Date')
    plt.ylabel('Normalized Value')
    plt.grid(True)
    plt.show()

def calculate_return(data):
    # NORMALIZATION - to measure all variables in comparable metrics
    log_return = np.log(data/data.shift(1))     # S(t)/S(t-1)
    correlation_matrix = log_return.corr()
    print(correlation_matrix)
    return log_return[1:] 

def show_correlation_matrix(log_return):
    corr_matrix = log_return.corr()
    plt.figure(figsize=(8,6))
    plt.imshow(corr_matrix, cmap = 'coolwarm', vmin =0, vmax =1)
    plt.colorbar(label = 'Correlation')
    plt.xticks(range(len(corr_matrix.columns)),corr_matrix.columns, rotation = 45)
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation = 45)
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            plt.text(j, i, f'{corr_matrix.iloc[i,j]: .2f}', ha= 'center', va = 'center', color = 'black')
    plt.title('Correlation Matrix of Asset Log-Returns')
    plt.tight_layout()
    plt.show()


def show_statistics(returns):
    #instead of daily metrics we are after annual metrics 
    # mean of annual return 
    print(returns.mean() * NUM_TRADING_DAYS)
    print(returns.cov() * NUM_TRADING_DAYS)

def show_mean_variance(returns, weights):
    #annual return
    portfolio_returns = np.sum(returns.mean() * weights * NUM_TRADING_DAYS)
    portfolio_volatility = np.sqrt(np.dot(weights.T,np.dot(returns.cov() * NUM_TRADING_DAYS, weights)))
    print('Expected portfolio mean (return): ', portfolio_returns)
    print('Expected portfolio volatility (standard deviation): ', portfolio_volatility)

def show_portfolio(returns, volatilities):
    plt.figure(figsize=(10,6))
    sharpe = (returns - RISK_FREE)/volatilities
    plt.scatter(volatilities, returns, c=sharpe, marker='o')
    plt.grid(True)
    plt.xlabel('Expected volatility')
    plt.ylabel('Excpected return')
    plt.colorbar(label= 'Sharpe Ratio')
    plt.title('Simulated portofolio')
    plt.show()


def generate_portfolio(returns):
    portfolio_means = []
    portfolio_risk = []
    portfolio_weights = []

    for _ in range (num_portfolio):
        w = np.random.random(len(stocks))
        w /= np.sum(w)
        portfolio_weights.append(w)
        portfolio_means.append(np.sum(returns.mean() * w) *NUM_TRADING_DAYS)
        portfolio_risk.append(np.sqrt(np.dot(w.T, np.dot(returns.cov()*NUM_TRADING_DAYS, w))))

    return np.array(portfolio_weights),np.array(portfolio_means), np.array(portfolio_risk)


def statistics (weights, returns):
    portfolio_return=(np.sum(returns.mean() * weights) *NUM_TRADING_DAYS)
    portfolio_volatility = (np.sqrt(np.dot(weights.T, np.dot(returns.cov()*NUM_TRADING_DAYS, weights))))
    sharpe_ratio = (portfolio_return - RISK_FREE) / portfolio_volatility
    return np.array([portfolio_return, portfolio_volatility,sharpe_ratio])

#scipy optimize module can find the minimun of a given function
#the maximum of a f(x) is the minimun of -f(x)
def min_function_sharpe(weights, returns):
    return -statistics(weights, returns)[2]

def optimize_portfolio(weights, returns):
    constraints = {'type': 'eq', 'fun' : lambda x: np.sum(x) -1 } #the sum of weights == 1
    #the weights can be 1 at most: 1 when 100% of money is invested into a single stock
    bounds = tuple((0,1) for _ in range(len(stocks)))
    return optimization.minimize(fun=min_function_sharpe, x0= weights[0], args=(returns), method='SLSQP', bounds= bounds, constraints=constraints)

def print_optimal_portfolio(optimun, returns):
    print('Optimal portfolio: ', optimun['x'].round(3))
    print('Expected return, volatility adn Sharpe ratio: ', statistics(optimun['x'].round(3), returns))

def minimize_volatility(weights, returns):
    return statistics(weights, returns)[1]


def optimize_min_variance(weights, returns):
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(len(stocks)))
    return optimization.minimize(fun=minimize_volatility,x0=weights[0],args=(returns,),method='SLSQP',bounds=bounds,constraints=constraints)



def show_optimal_portfolio(max_sharpe, min_var, rets, portfolio_rets, portfolio_vols):
    plt.figure(figsize=(10, 6))
    sharpe = (portfolio_rets - RISK_FREE) / portfolio_vols
    plt.scatter(portfolio_vols,portfolio_rets,c=sharpe,marker='o',alpha=0.7)
    plt.colorbar(label='Sharpe Ratio')
    max_sharpe_stats = statistics(max_sharpe['x'], rets)
    min_var_stats = statistics(min_var['x'], rets)
    plt.scatter(max_sharpe_stats[1],max_sharpe_stats[0],marker='*',s=350,color='green',label='Max Sharpe Portfolio')
    plt.scatter(min_var_stats[1],min_var_stats[0],marker='X',s=250,color='red',label='Minimum Variance Portfolio')
    plt.grid(True)
    plt.xlabel('Expected volatility')
    plt.ylabel('Expected return')
    plt.title('Markowitz Portfolio Optimization')
    plt.legend()
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':

    dataset = download_data()
    show_data(dataset)    
    log_daily_returns = calculate_return(dataset)
    show_correlation_matrix(log_daily_returns)
    pweights, means, risks = generate_portfolio(log_daily_returns)
    show_portfolio(means, risks)
    max_sharpe = optimize_portfolio(pweights, log_daily_returns)
    min_var = optimize_min_variance(pweights, log_daily_returns)
    print('Max Sharpe Portfolio')
    print_optimal_portfolio(max_sharpe, log_daily_returns)
    print('\nMinimum Variance Portfolio')
    print_optimal_portfolio(min_var, log_daily_returns)
    show_optimal_portfolio(max_sharpe,min_var,log_daily_returns,means,risks)
