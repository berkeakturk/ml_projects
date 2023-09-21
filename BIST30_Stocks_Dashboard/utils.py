import requests
from bs4 import BeautifulSoup
import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import date

def get_stock_names(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        str_soup = str(soup)
        stock_names = []
        counter = 0
        first_cut_str = '<div class="vcell">BIST 30</div>'
        last_cut_str = '<div class="column-type1 wide vtable offset" id="115">'
        substr_end = '</a>\n</div>\n<div class="comp-cell _03 vtable">\n<a'
        corpus = str_soup[str_soup.find(first_cut_str) : str_soup.find(last_cut_str)]
        while counter < 30:
            finish_index = corpus.find(substr_end)
            temp_substr = corpus[:finish_index]
            stock_name = temp_substr[temp_substr.rfind('>')+1:]
            stock_name = stock_name + '.IS'
            stock_names.append(stock_name)
            counter += 1
            corpus = corpus[finish_index+1:]
    else:
        st.write(f"Response Status Code = f{response.status_code}")
        return -1
    return stock_names

def get_all_stock_prices(stock_names, start_date, end_date, interval="1d"):
    df_stocks = yf.download(stock_names, start=start_date, end=end_date, interval=interval)
    df_stocks = df_stocks.Close
    df_stocks.index = df_stocks.index.strftime('%Y-%m-%d')
    return df_stocks

def get_stock_prices(stock_names, start_date, end_date, interval="1d"):
    df = yf.download(stock_names, start=start_date, end=end_date, interval=interval)
    df.index = df.index.strftime('%Y-%m-%d')
    return pd.DataFrame(df['Close'])

def color_negative_red(val):
    color = 'red' if val < 0 else 'green'
    return f'color: {color}'

def get_price_changes(df, interval=1):
    df_ = (df - df.shift(interval)) / df.shift(interval)
    df__ = df_.tail(1).T
    df__.columns = [f'{interval}d Pct Change']
    df__ = df__.sort_values(by=f'{interval}d Pct Change', ascending=False)
    styled_df = df__.style.format({f'{interval}d Pct Change': '{:.2%}'})
    styled_df = styled_df.applymap(color_negative_red)
    return styled_df

def get_earnings_within_interval(df, start_date, end_date):
    returns = (df[df.index <= end_date].tail(1).iloc[0].values - df[df.index >= start_date].head(1).iloc[0].values) / df[df.index >= start_date].head(1).iloc[0].values
    temp_dict = {'Stocks': df.columns, 'Pct Change': returns}
    df_ = pd.DataFrame(temp_dict)
    df_ = df_.sort_values(by='Pct Change', ascending=False)
    df_ = df_.set_index('Stocks')
    styled_df = df_.style.format({'Pct Change': '{:.2%}'})
    styled_df = styled_df.applymap(color_negative_red)
    return styled_df


