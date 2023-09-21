import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st


def plot_stock_prices(df):
    for stock in sorted(list(df.columns)):
        fig = px.line(df, x=df.index, y=stock, title = f'{stock} Prices')
        fig.update_xaxes(title_text='Date')
        fig.update_yaxes(title_text='Price')
        st.plotly_chart(fig, use_container_width=True)

def plot_pct_change(df_pct):
    for stock in sorted(list(df_pct.columns)):
        fig = px.line(df_pct, x=df_pct.index, y=stock, title=f'{stock} Percentage Change')
        fig.update_xaxes(title_text='Date')
        fig.update_yaxes(title_text='Percentage Change')
        st.plotly_chart(fig, use_container_width=True)

def plot_stock_probs(df):
    df_ = df.sort_values(by='Probability', ascending=False)
    df_['Probability'] *= df_['Probability'] * 100
    fig = px.bar(df_, x=df_.index, y='Probability', title="Stock Probability Bar Chart")
    fig.update_xaxes(title_text='Stock Name')
    fig.update_yaxes(title_text='Probability for Uplift (%)')
    st.plotly_chart(fig, use_container_width=True)