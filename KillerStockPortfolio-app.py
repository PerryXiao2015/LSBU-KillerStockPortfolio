# https://www.youtube.com/watch?v=bvDkel5whUY

import pandas as pd
import numpy as np
import requests
#!pip install yfinance
import yfinance as yf
from datetime import date
import datetime as dt
import streamlit as st
#!pip install PyPortfolioOpt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
#pip install pulp
# Get the discret allocation of each stock
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

#pip install cvxpy
#pip install cvxopt

st.title('Killer Stock Portfolio App')

st.markdown("""
This app retrieves the list of the **S&P 500** and **FTSE 100** from Wikipedia. Then gets the corresponding **stock closing price** , and generate a killer portfolio with fund allocation!
* **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, seaborn, yfinance
* **Data source:** [Wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies).
""")

st.sidebar.header('User Input Features')

# Web scraping of S&P 500 data
#
@st.cache
def load_data():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    html = pd.read_html(url, header = 0)
    df = html[0]
    return df

@st.cache
def load_ftse100data():
    url = 'https://en.wikipedia.org/wiki/FTSE_100_Index'
    html = pd.read_html(url, header = 0)
    df = html[3]
    return df

# Download stock data from Yahoo Finance
#
@st.cache
def get_data(symbols):
    symbols2 =[]
    t = []
    today = date.today()
    # End date of the stock
    d1 = today.strftime("%Y-%m-%d")

    d = periods*30
    d0 = date.today()-dt.timedelta(days=d)
    
    # Start date of the stock
    #d0 = '2010-5-31'
    #print("d1 =", d1)
    l = -1 
    # get all the data 
    tDf = pd.DataFrame()

    for tickerSymbol in symbols:
      #get data on this ticker
      tickerData = yf.Ticker(tickerSymbol)
      
      #print(tickerData)
      #get the historical prices for this ticker
      #tickerDf = tickerData.history(period='1d', start='2010-5-31', end='2021-8-31')
      tickerDf = tickerData.history(period='1d', start= d0, end= d1)
      
      #print(tickerDf.empty)
      if not tickerDf.empty:
          #tDf.append(tickerDf.Close) 
          #tDf.append(tickerDf['Close'].values)
          if tDf.empty:
              tDf = pd.DataFrame(tickerDf.Close)
              #print(tDf)
          else:
              #tDf = pd.merge(tDf, pd.DataFrame(tickerDf.Close), left_index=True, right_index=True)
              tDf = pd.concat([tDf, pd.DataFrame(tickerDf.Close)], axis=1)
              #tDf.join(pd.DataFrame(tickerDf.Close))
              #print(pd.DataFrame(tickerDf.Close))
              #print(tDf)
          symbols2.append(tickerSymbol)
          if len(tickerDf.index)>l:
              t = tickerDf.index
              l = len(t) 
              print(l)

    df = tDf

    #df = pd.DataFrame(tDf).T
    df.columns = symbols2
    #df = df.set_index(pd.DatetimeIndex(t))
    return df

# Calculate the stock portfolio from Yahoo Finance stock data
#
@st.cache
def get_portfolio(tdf):
    assets = tdf.columns
    print(assets)
    mu = expected_returns.mean_historical_return(tdf)
    S = risk_models.sample_cov(tdf)
    
    ef = EfficientFrontier(mu,S)
    weights = ef.max_sharpe()

    cleaned_weights = ef.clean_weights()
    #print(cleaned_weights)
    pf = ef.portfolio_performance(verbose = True)
    return cleaned_weights, pf
    
# Calculate the stock portfolio from Yahoo Finance stock data
#
@st.cache
def get_allocation(cleaned_weights, tdf):
    # The amount of money you want to invest ($)
    #portfolio_val = 10000
    latest_prices = get_latest_prices(tdf)
    weights = cleaned_weights
    da = DiscreteAllocation(weights,latest_prices, total_portfolio_value = portfolio_val)
    allocation, leftover = da.lp_portfolio()
    print('Discrete allocation:', allocation)
    print('Funds Remaining: $', leftover)
    return allocation, leftover


source = st.sidebar.selectbox(
     'Select a Stock Market:',
     ["S&P 500","FTSE 100"], index=0)
#st.sidebar.write('You selected:', source)
if source == "S&P 500":
    df = load_data()
    symbols = df['Symbol'].values
    #symbols = ['MMM', 'ABT', 'ABBV', 'ABMD']
    #print(df)
    #coms = df.groupby('EPIC')

    #coms = df['Symbol'].unique()
    #names = df['Security'].unique() 
    coms = df['Symbol']
    names = df['Security']
    #print(coms)

    # Sidebar - Sector selection
    #selected_scoms = st.sidebar.multiselect('Companies', coms , coms )

    option1 = st.sidebar.selectbox(
         'Select a Company:',
         coms, index=0)
    #st.sidebar.write('You selected:', option1)
    i=[i for i, j in enumerate(coms) if j == option1]
    #i = np.where(coms == option1)
    name = names[i]
    #print(i)
    #print(name)
    tickerSymbol = option1
    #st.sidebar.write(option1)

elif source == "FTSE 100":
    df = load_ftse100data()
    symbols = df['Company'].values
    #print(df)
    #coms = df.groupby('EPIC')

    coms = df['EPIC']
    names = df['Company']
    #print(coms)

    # Sidebar - Sector selection
    #selected_scoms = st.sidebar.multiselect('Companies', coms , coms )

    option2 = st.sidebar.selectbox(
         'Select a Company:',
         coms, index=0)
    #st.sidebar.write('You selected:', option2)
    i=[i for i, j in enumerate(coms) if j == option2]
    #i = np.where(coms == option2)
    name = names[i]
    tickerSymbol = option2
    #st.sidebar.write(option2)

#Checkbox for Periods
selectperiod = st.sidebar.selectbox("Select Period", [ "1 Year","5 Years", "10 Years", "3 Months", "6 Months"])
periods = 12
if selectperiod == "1 Year":
    periods = 12
elif selectperiod == "5 Years":
    periods = 60
elif selectperiod == "10 Years":
    periods = 120
elif selectperiod == "3 Months":
    periods = 3
elif selectperiod == "6 Months":
    periods = 6

#Checkbox for Investment
selectval = st.sidebar.selectbox("Select Amount of Investment [$]", ["5000", "10000", "20000"])
portfolio_val = 10000
if selectval == "5000":
    portfolio_val = 5000
elif selectval == "10000":
    pportfolio_val = 10000
elif selectval == "20000":
    portfolio_val = 20000
    
#st.header('Display Companies')   
if st.sidebar.button('Get Portforlio and Allocation'):
    st.header('Killer Stock Portfolio and Fund Allocation $'+ str(portfolio_val))
    tdf = get_data(symbols)
    st.write(tdf)
    cleaned_weights, pf = get_portfolio(tdf)
    st.markdown('**Killer Portfolio:**')
    st.write(' Weights: ')
    st.write(cleaned_weights)
    st.write(' Expected annual return: '+ str(pf[0]*100) + '%')
    st.write(' Annual volatility:      '+ str(pf[1]*100) + '%')
    st.write(' Sharpe Ratio:           '+ str(pf[2]*100) + '%')

    st.markdown('**Fund Allocation:**')
    allocation, leftover = get_allocation(cleaned_weights, tdf)
    st.write(' Discrete allocation:    '+ str(allocation))
    st.write(' Funds Remaining:       $'+ str(leftover))


