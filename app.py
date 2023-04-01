import numpy as np
import pandas as pd
import yfinance as yf
import plotly.tools as tls
from datetime import date
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

class StockPriceForecasting:
    
    def __init__(self, start_date='2010-01-01', end_date=date.today().strftime("%Y-%m-%d")):
        self.start_date = start_date
        self.end_date = end_date
        
    def download_data(self, ticker):
        self.df = yf.download(ticker, start=self.start_date, end=self.end_date)
    
    def describe_data(self):
        year = self.end_date[0:4]
        st.subheader("Data from 2010 - {}".format(year))
        st.write(self.df.describe())
    
    def plot_closing_price_vs_time(self):
        fig = plt.figure(figsize=(12,6))
        plt.plot(self.df.Close, label='Close')
        plotly_fig = tls.mpl_to_plotly(fig)
        leg = plt.legend()
        plotly_fig.update_layout(width=1000, height=1000)
        plotly_fig.data[0].line.color = "orangered"
        st.plotly_chart(plotly_fig, width=1000, height=1000)
    
    def plot_closing_price_vs_time_with_ma(self, window_size=100):
        ma = self.df.Close.rolling(window_size).mean()
        fig_ma = plt.figure(figsize=(12,6))
        plt.plot(ma, label=f'{window_size}MA')
        plt.plot(self.df.Close , label="Closing Price")
        plotly_fig = tls.mpl_to_plotly(fig_ma)
        leg = plt.legend()
        st.plotly_chart(plotly_fig, width=1000, height=1000)
    
    def plot_closing_price_vs_time_with_ma_and_ema(self, ma_window_size=100, ema_window_size=200):
        ma = self.df.Close.rolling(ma_window_size).mean()
        ema = self.df['Close'].ewm(span=ema_window_size, adjust=False).mean()
        fig_ma_ema = plt.figure(figsize=(12,6))
        plt.plot(self.df.Close, label="Closing Price")
        plt.plot(ma, label=f'{ma_window_size}MA')
        plt.plot(ema, label=f'{ema_window_size}EMA')
        plotly_fig = tls.mpl_to_plotly(fig_ma_ema)
        plotly_fig.data[0].line.color = "teal"
        plotly_fig.data[1].line.color = "white"
        plotly_fig.data[2].line.color = "red"
        leg = plt.legend()
        st.plotly_chart(plotly_fig, width=1000, height=1000)
    
    def predict_stock_price(self, ticker, model_file_path='keras_stock1.h5', train_test_split_ratio=0.7, window_size=100):
        self.download_data(ticker)
        data_trainig = pd.DataFrame(self.df['Close'][0:int(len(self.df)*train_test_split_ratio)])
        data_testing = pd.DataFrame(self.df['Close'][int(len(self.df)*train_test_split_ratio):int(len(self.df))])
        scaler = MinMaxScaler(feature_range=(0,1))
        data_trainig_array = scaler.fit_transform(data_trainig)
        model = load_model(model_file_path)
        past100 = data_trainig.tail(window_size)
        final_df = past100.append(data_testing, ignore_index=True)

        

start = '2010-01-01'
end = date.today().strftime("%Y-%m-%d")

st.title("Stock Price Forecasting")

user_input = st.text_input("Enter Stock Ticker","BHARTIARTL.NS")

spf = StockPriceForecasting(user_input, start, end)

spf.describe_data()

spf.plot_closing_price()

spf.plot_closing_price_ma(100)

spf.plot_closing_price_ma(100, 200)

spf.plot_closing_price_ema(100, 200)

spf.plot_predicted_vs_actual()
        
