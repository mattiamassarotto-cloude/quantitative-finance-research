import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import statsmodels.api as sm

#market interest rate

#we will consider monthly returns - and we want to calculate the annual return 
MONTHS_IN_YEAR = 12

class CAPM:
    def __init__(self, stocks, start_date, end_date):
        self.data = None 
        self.stocks = stocks
        self.start_date = start_date
        self.end_date = end_date
        RISK_FREE= yf.download('^TNX', start= self.start_date, end= self.end_date, auto_adjust= True)['Close']
        self.RISK_FREE_RATE = RISK_FREE.mean().item() / 100 
        

    def download_data(self):
        data = {}
        for stock in self.stocks:
            ticker = yf.Ticker(stock)
            data[stock] = ticker.history(start= self.start_date, end= self.end_date, auto_adjust = True)['Close']
        return pd.DataFrame(data)
    
    def initialize(self):
        stock_data = self.download_data()
        #use monthly returns instead of daily returns
        stock_data = stock_data.resample('ME').last()
        self.data = pd.DataFrame ({'s_adjclose': stock_data[self.stocks[0]],
                                'm_adjclose': stock_data[self.stocks[1]]})
        # logarithmic montly returns
        self.data[['s_returns','m_returns']] = np.log (self.data[['s_adjclose','m_adjclose']]/
                                                    self.data[['s_adjclose','m_adjclose']].shift(1))
        
        rf_monthly = np.log(1+self.RISK_FREE_RATE)/MONTHS_IN_YEAR
        self.data['s_excess'] = self.data['s_returns'] - rf_monthly
        self.data['m_excess'] = self.data['m_returns'] - rf_monthly
        self.data = self.data[1:]
        print(self.data)
        print(f'risk free rate = '+ str(self.RISK_FREE_RATE))

    def calculate_beta(self): #how risky ur portfolio relative to the market
        #covariance matrix: the diagonal items are the variances
        #off diagonal are the covariance
        #the matrix is symmetric: cov[0,1] = cov[1,0]
        covariance_matrix = np.cov(self.data['s_excess'], self.data['m_excess'])
        #calcuate beta according to the formula
        beta = covariance_matrix[0,1] / covariance_matrix[1,1]
        print('beta from formula', beta)
        #beta = 1 stock moving exacty with the market has beta 1
        #beta > 1 stock market risk is higher than that of an average stock
        #beta < 1 stock market risk is lower than that of an average stock


    def regression(self):
        x = self.data['m_excess'].replace([np.inf, -np.inf], np.nan)
        y = self.data['s_excess'].replace([np.inf, -np.inf], np.nan)
        mask = x.notna() & y.notna()
        X = sm.add_constant(x[mask])
        model = sm.OLS(y[mask], X).fit()
        alpha = model.params.iloc[0]
        beta = model.params.iloc[1]
        print(model.summary())
        print(f'Alpha: {alpha:.6f}')
        print(f'Beta: {beta:.6f}')
        print(f'R²: {model.rsquared:.6f}')
        print(f'p-value beta: {model.pvalues.iloc[1]:.6f}')
        self.plot_regression(alpha, beta, model.rsquared)


    def plot_regression(self, alpha, beta, r2):
        fig, axis = plt.subplots(figsize=(12, 7))
        x = self.data['m_returns']
        y = self.data['s_returns']
        axis.scatter(x,y,color = 'black',alpha=0.75,s=45,label='Osservazioni mensili')
        x_line = np.linspace(x.min(), x.max(), 200)
        y_line = beta * x_line + alpha
        axis.plot(x_line,y_line,linewidth=2.5,label='Retta CAPM stimata')
        axis.axhline(0, linewidth=0.8)
        axis.axvline(0, linewidth=0.8)
        axis.set_title('CAPM: relazione tra rendimenti mensili di JPMorgan e S&P 500',fontsize=15)
        axis.set_xlabel('Rendimento mensile del mercato (S&P 500)',fontsize=13)
        axis.set_ylabel('Rendimento mensile di JPMorgan',fontsize=13)
        stats_text = (f'α = {alpha:.4f}\n'f'β = {beta:.4f}\n'f'R² = {r2:.4f}')
        axis.text(0.03,0.93,stats_text,transform=axis.transAxes,fontsize=11,verticalalignment='top',bbox=dict(boxstyle='round',facecolor='white',alpha=0.9))
        axis.grid(True, alpha=0.35)
        axis.legend(loc='lower right')
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    capm = CAPM(['JPM', '^GSPC'], '2005-01-01', '2012-12-31')
    capm.initialize()
    capm.calculate_beta()
    capm.regression()
    