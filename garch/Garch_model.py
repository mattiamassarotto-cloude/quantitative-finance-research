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
            dist='t'
        )

        result = model.fit(update_freq=5)
        print(result.summary())

        vol = result.conditional_volatility
        print("Volatilità massima:")
        print(vol.max())

        print("Data volatilità massima:")
        print(vol.idxmax())

        max_vol = vol.max()
        max_date = vol.idxmax()

        plt.figure(figsize=(12, 6))
        plt.plot(vol, label='Volatilità condizionata')
        plt.axvspan('2008-09-01','2009-03-01',alpha=0.2,color='red',label='Crisi Lehman Brothers')
        plt.scatter(max_date,max_vol,color='red',s=80,zorder=5)
        plt.annotate(f'Max vol = {max_vol:.2f}',xy=(max_date, max_vol),xytext=(40, -25),textcoords='offset points',
            arrowprops=dict(arrowstyle='->'),
            fontsize=10)
        plt.title('Volatilità condizionata stimata mediante GARCH(1,1)')
        plt.xlabel('Data')
        plt.ylabel('Deviazione standard condizionata (%)')
        plt.legend()
        plt.grid(True)
        plt.show()


        # Parametro nu stimato dal GARCH
        nu = result.params["nu"]

        # Asse X
        x = np.linspace(-6, 6, 1000)

        # Densità
        normal_pdf = norm.pdf(x)
        student_pdf = t.pdf(x, df=nu)


        # Grafico classico
        plt.figure(figsize=(12,6))

        plt.plot(
            x,
            normal_pdf,
            label="Distribuzione normale",
            linewidth=2)

        plt.plot(
            x,
            student_pdf,
            label=f"t-Student, ν = {nu:.2f}",
            linewidth=2)

        plt.title("Confronto tra distribuzione normale e t-Student")
        plt.xlabel("Shock standardizzato")
        plt.ylabel("Densità")
        plt.legend()
        plt.grid(True)

        plt.show()

        # Grafico professionale
        # Scala logaritmica

        plt.figure(figsize=(12,6))

        plt.semilogy(
            x,
            normal_pdf,
            label="Distribuzione normale",
            linewidth=2)
        
        plt.semilogy(
            x,
            student_pdf,
            label=f"t-Student, ν = {nu:.2f}",
            linewidth=2)

        plt.title("Confronto delle code: Normale vs t-Student (scala logaritmica)")
        plt.xlabel("Shock standardizzato")
        plt.ylabel("Densità (scala logaritmica)")
        plt.legend()
        plt.grid(True, which="both", linestyle="--")

        plt.show()

        print("Normale:")
        print(2 * (1 - norm.cdf(4)))       
        print("t-Student:")
        print(2 * (1 - t.cdf(4, df=nu)))



        return result

if __name__ == '__main__':
    asset = STOCK('^GSPC', '2005-01-01', '2012-12-31')
    result = asset.garch_model()