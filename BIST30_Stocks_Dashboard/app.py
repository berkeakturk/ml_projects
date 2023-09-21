import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import date
from utils import get_stock_names, get_stock_prices, get_all_stock_prices, get_price_changes, get_earnings_within_interval
from plots import plot_stock_prices, plot_pct_change, plot_stock_probs
from modeling import DistributionAnalyzer

st.set_page_config(layout='wide')

url = "https://www.kap.org.tr/tr/Endeksler"
stock_names = get_stock_names(url)
stock_names.remove('ASTOR.IS')

with st.sidebar:
    st.header('BIST 30 Stock Price Dashboard')

    stocks = st.multiselect(
    'Select the stock',
    stock_names)

    start_date = st.date_input(
    "Select a starting date", 
    date(2023, 1, 1))

    end_date = st.date_input(
    "Select an ending date",
    date.today())

    is_summary = st.checkbox('Add Summary for Last 1-7-14-28 Days')

    is_pct_change = st.checkbox('Add Percentage Change')

    is_calculate_prob = st.checkbox('Calculate Probabilities')
    if is_calculate_prob:
        uplift = st.number_input("Enter the Uplift To Calculate the Probability",
                                       min_value=-100.0,
                                       max_value=100.0,
                                       step=0.01,
                                       value=2.0)
        uplift /= 100


    #is_ma = st.checkbox('Add Moving Average')
    #if is_ma:
    #    ma_option = st.slider('Moving Average Interval', 1, 30, 7)

    start_date, end_date = pd.Timestamp(start_date), pd.Timestamp(end_date)
    stocks = sorted(stocks)

if st.sidebar.button("Run"):
    df_all = get_all_stock_prices(sorted(stock_names), start_date, end_date)
    if is_summary:
        st.header('BIST 30 Summary')
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.write('1D Earnings')
            st.write(get_price_changes(df_all, interval=1))
        with col2:
            st.write('7D Earnings')
            st.write(get_price_changes(df_all, interval=7))
        with col3:
            st.write('14D Earnings')
            st.write(get_price_changes(df_all, interval=14))
        with col4:
            st.write('28D Earnings')
            st.write(get_price_changes(df_all, interval=28))
        with col5:
            st.write(f"Earnings from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            st.write(get_earnings_within_interval(df_all, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
    
    if is_calculate_prob:
        st.header(f'Probability to Exceed the Inputted Uplift = {uplift*100}%')
        dist_analyzer = DistributionAnalyzer(df_all)
        prob_df = dist_analyzer.calculate_probabilities(df_all, uplift)
        plot_stock_probs(prob_df)

    if len(stocks) > 0:
        df = get_stock_prices(stocks, start_date, end_date)
        df.columns = stocks

        if is_pct_change:
            st.header("Selected Stocks' Price And Percentage Change Graphs")
            df_pct = df.pct_change().dropna() * 100
            col1, col2 = st.columns(2)
            with col1:
                plot_stock_prices(df)
            with col2:
                plot_pct_change(df_pct)
        else:
            st.header("Selected Stocks' Price Graphs")
            plot_stock_prices(df)


