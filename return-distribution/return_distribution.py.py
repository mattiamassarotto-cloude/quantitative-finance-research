import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm, skew, kurtosis, t


def download_data (stock, start_date, end_date):
    data = {}
    ticker = yf.download(stock, start = start_date, end = end_date, auto_adjust= False)
    price = ticker['Close']
    if 'Adj Close' in ticker.columns:
        price = ticker['Adj Close']
    else:
        price = ticker['Close']
    
    if isinstance(price, pd.DataFrame):
        price = price.iloc[:,0]
    
    return price.rename('Price').to_frame()
    
def calculate_returns(stock_data):
    stock_data = stock_data.copy()
    stock_data['LogReturn'] = np.log(stock_data['Price'] / stock_data['Price'].shift(1))
    return stock_data[1:]

def show(stock_data):
    plt.hist(stock_data, bins=40, density=True, alpha = 0.7,color="lightgray",edgecolor='black', label= 'Empirical Distribution') #stock_data → i rendimenti (log-returns) bins=700 → numero di “cassette” in cui dividi i dati
    stock_variance = stock_data.var() #Calcola la varianza campionaria dei rendimenti: Var(R)=E[(R−μ)2] = volatilita' al quadrato
    stock_mean = stock_data.mean() #Calcola la media campionaria: μ=E[R] = È il rendimento medio giornaliero.
    sigma = np.sqrt(stock_variance) #Radice quadrata della varianza → deviazione standard: σ=Var(R) = È la volatilità giornaliera:
    x = np.linspace(stock_mean - 6 * sigma, stock_mean + 6 * sigma, 2000)#Crea un array di 100 punti equispaziati tra: [μ−5σ, μ+5σ]
                                                                        #Perché ±5σ? N(μ,σ²) contiene: 
                                                                        #99.7% entro ±3σ 
                                                                        # ±5σ copre praticamente tutta la massa
                                                                        #Serve per disegnare la curva liscia.
    
    plt.plot(x, norm.pdf(x, stock_mean, sigma), 
            linewidth = 2.5, color='red', label = 'Normal distribution') #Disegna la densità di probabilità della normale: ​norm.pdf = Probability Density Function, usa la media e la volatilità stimate dai dati
                                                            #Distribuzione empirica dei rendimenti VS Modello teorico normale                         
    student_pdf = t.pdf((x - stock_mean) / sigma,df=4) / sigma #t-student
    plt.plot(x,student_pdf,linewidth = 2.5,color = 'blue', label= 'Student-t (df=4)')    
    plt.xlabel('Log-returns')
    plt.ylabel("Density")
    plt.legend()
    plt.title("Distribution of log-returns vs Normal")
    plt.show()

def show_log_tails(log_returns):
    plt.hist(log_returns, bins=40, density=True, alpha = 0.7,color="lightgray",edgecolor='black', label= 'Empirical Distribution')
    stock_variance = log_returns.var() 
    stock_mean = log_returns.mean()
    sigma = np.sqrt(stock_variance) 
    x = np.linspace(stock_mean - 6 * sigma, stock_mean + 6 * sigma, 2000)
                
    
    plt.plot(x, norm.pdf(x, stock_mean, sigma), 
            linewidth = 2.5, color='red', label = 'Normal distribution')
                                                                                
    student_pdf = t.pdf((x - stock_mean) / sigma,df=4) / sigma 
    plt.plot(x,student_pdf,linewidth = 2.5,color = 'blue', label= 'Student-t (df=4)')

    plt.yscale('log')
    plt.xlabel('Log-returns')
    plt.ylabel("Density (log scale)")
    plt.legend()
    plt.title("Fat tails analysis")
    plt.show()





if __name__ == '__main__':
    stock_data= download_data('JPM', '2005-01-01', '2012-12-31')
    returns = calculate_returns(stock_data)
    print(returns.head())
    log_returns = returns['LogReturn']
    skewness = skew(log_returns)
    kurt = kurtosis(log_returns, fisher=False) #fisher = False --> normale = 3
    
    print("Statistiche dei rendimenti:")
    print(f"Mean: {log_returns.mean():.5f}")
    print(f"Std Dev: {log_returns.std():.5f}")
    print(f"Skewness: {skewness:.4f}")
    print(f"Kurtosis: {kurt:.4f}")
    show(log_returns)
    show_log_tails(log_returns)
