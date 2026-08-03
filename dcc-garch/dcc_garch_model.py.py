import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as sgt
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats.distributions import chi2
import statsmodels.tsa.stattools as sts 
import seaborn as sns
import statsmodels.api as sm
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from docx import Document
from docx.shared import Pt
sns.set()
from statsmodels.tsa.stattools import coint
from arch import arch_model
from scipy.stats import norm, t 


stocks = ['AAPL', 'JPM', 'XOM', 'MSFT', 'JNJ']

class STOCK:
    def __init__(self, stock, start_date, end_date):
        self.data = None 
        self.stock = stock
        self.start_date = start_date
        self.end_date = end_date

    def download_data(self):
        ticker = yf.Ticker(self.stock)
        data = ticker.history(
            start=self.start_date,
            end=self.end_date,
            auto_adjust=True
        )['Close']
        return data.dropna()

    def log_returns(self):
        prices = self.download_data()
        returns = np.log(prices / prices.shift(1)).dropna()
        return returns * 100  
    
    def garch_model(self):
        returns = self.log_returns()

        model = arch_model(
            returns,
            mean='Constant',
            vol='GARCH',
            p=1,
            q=1,
            dist='t')

        result = model.fit(update_freq=5, disp= 'off')
        standardized_residuals = (result.resid / result.conditional_volatility)

        return standardized_residuals


def dcc_garch(std_resid, a=0.05, b=0.93,asset_1 ='AAPL', asset_2='JPM'):

    Q_bar = std_resid.corr().values
    Q_t = Q_bar.copy()
    correlations = []
    for t in range(1, len(std_resid)):
        u = std_resid.iloc[t-1].values.reshape(-1, 1)
        Q_t = ((1 - a - b) * Q_bar + a * (u @ u.T) + b * Q_t)
        D_inv = np.diag(1 / np.sqrt(np.diag(Q_t)))
        R_t = D_inv @ Q_t @ D_inv
        correlations.append(R_t)

    i = list(std_resid.columns).index(asset_1)
    j = list(std_resid.columns).index(asset_2)

    dynamic_corr = pd.Series([matrix[i, j] for matrix in correlations],index=std_resid.index[1:])

    max_corr = dynamic_corr.max()
    max_date = dynamic_corr.idxmax()
    mean_corr_value = dynamic_corr.mean()

    plt.figure(figsize=(12, 6))
    plt.plot(dynamic_corr.index, dynamic_corr)
    plt.title(f"Correlazione condizionata dinamica tra AAPL e JPM (DCC-GARCH)")
    plt.xlabel("Date")
    plt.ylabel("Correlation")
    plt.axvspan('2008-09-15','2009-03-01',alpha=0.2,color='red',label='Crisi Lehman Brothers')
    plt.axhline(mean_corr_value,linestyle='--',color='black',alpha=0.7,label=f'Correlazione media = {mean_corr_value:.2f}')
    plt.scatter(max_date,max_corr,color='red',s=70,zorder=5)
    plt.annotate(f'Max corr = {max_corr:.2f}',xy=(max_date, max_corr),xytext=(40, -25),textcoords='offset points',
        arrowprops=dict(arrowstyle='->'),fontsize=10)
    plt.legend()
    plt.grid(True)
    plt.show()


    mean_portfolio_corr =[]
    for matrix in correlations:
        n = matrix.shape[0]
        values =[]
        for i in range(n):
            for j in range (i+1, n):
                values.append(matrix[i,j])
        mean_portfolio_corr.append(np.mean(values))

    mean_portfolio_corr_value = np.mean(mean_portfolio_corr)

    max_corr = max(mean_portfolio_corr)
    max_idx = np.argmax(mean_portfolio_corr)
    max_date = std_resid.index[1:][max_idx]

    plt.figure(figsize=(12,6))
    plt.plot(std_resid.index[1:],mean_portfolio_corr,linewidth=1.5)
    plt.axvspan('2008-09-15','2009-03-31',alpha=0.2,color='red',label='Crisi Lehman Brothers')
    plt.axhline(mean_portfolio_corr_value,linestyle='--',color='black',alpha=0.7,
        label=f'Correlazione media = {mean_portfolio_corr_value:.2f}')
    plt.scatter(max_date,max_corr,color='red',s=80,zorder=5)
    plt.annotate(f'Max corr = {max_corr:.2f}',xy=(max_date, max_corr),xytext=(30, -20),
        textcoords='offset points',arrowprops=dict(arrowstyle='->'))
    plt.title("Correlazione media dinamica del portafoglio stimata mediante DCC-GARCH")
    plt.xlabel("Date")
    plt.ylabel("Average Correlation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return correlations, dynamic_corr, mean_corr

if __name__ == '__main__':
        std_resid = pd.DataFrame()
        for stock in stocks:
            asset = STOCK(stock,'2000-01-01', '2012-12-31')
        
            resid = asset.garch_model()
            std_resid[stock] = resid
        dcc_corr, aapl_jpm_corr, mean_corr = dcc_garch(std_resid,asset_1='AAPL',asset_2='JPM')
