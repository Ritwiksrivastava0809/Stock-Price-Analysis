import numpy as np
import pandas as pd
import yfinance as yf
import plotly.tools as tls
from datetime import date
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

start = '2010-01-01'
end   = date.today().strftime("%Y-%m-%d")
st.title("Stock Price Forecasting")

user_input = st.text_input("Enter Stock Ticker","BHARTIARTL.NS")

df = yf.download(user_input,start=start,end=end)


# Describing Data
year = end[0:4]
st.subheader("Data from 2010 - {}".format(year))
st.write(df.describe())

# Visualaization
st.subheader("Closing Price vs Time chart")
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close ,label = 'Close')
plotly_fig = tls.mpl_to_plotly(fig)
leg = plt.legend()
plotly_fig.update_layout(width=1000, height=1000)
plotly_fig.data[0].line.color = "orangered"
st.plotly_chart(plotly_fig, width=1000, height=1000)

st.subheader("Closing Price vs Time chart with 100MA")
ma100 = df.Close.rolling(100).mean()
fig_ma100 = plt.figure(figsize=(12,6))
plt.plot(ma100,label = '100MA')
plt.plot(df.Close , label = "Closing Price")
plotly_fig = tls.mpl_to_plotly(fig_ma100)

leg = plt.legend()
st.plotly_chart(plotly_fig, width=1000, height=1000)


st.subheader("Closing Price vs Time chart with 100MA And 200MA")
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig_ma100_ma200 = plt.figure(figsize=(12,6))
plt.plot(df.Close , label = "Closing Price")
plt.plot(ma100,label = '100MA')
plt.plot(ma200 , label = '200MA')
plotly_fig = tls.mpl_to_plotly(fig_ma100_ma200)
leg = plt.legend()
plotly_fig.data[0].line.color = "brown"
plotly_fig.data[1].line.color = "white"
plotly_fig.data[2].line.color = "dark green"
st.plotly_chart(plotly_fig, width=1000, height=1000)


st.subheader("Closing Price vs Time chart with 100EMA And 200EMA")
EMA100 = df['Close'].ewm(span=100, adjust=False).mean()
EMA200= df['Close'].ewm(span=200, adjust=False).mean()
fig_ema100_ema200 = plt.figure(figsize=(12,6))
plt.plot(df.Close,label = "Closing Price")
plt.plot(EMA100,label = 'EMA100')
plt.plot(EMA200,label = 'EMA200')
leg = plt.legend()
plotly_fig = tls.mpl_to_plotly(fig_ema100_ema200)
leg = plt.legend()
plotly_fig.data[0].line.color = "teal"
plotly_fig.data[1].line.color = "white"
plotly_fig.data[2].line.color = "red"
st.plotly_chart(plotly_fig, width=1000, height=1000)


data_trainig = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

scaler = MinMaxScaler(feature_range=(0,1))
data_trainig_array = scaler.fit_transform(data_trainig)

model = load_model('keras_stock1.h5')

past100 = data_trainig.tail(100)
final_df = past100.append(data_testing,ignore_index=True)
input_data = scaler.fit_transform(final_df)
x_test = []
y_test = []

for i in range(100,input_data.shape[0]) :
    
    x_test.append(input_data[i-100 : i])
    y_test.append(input_data[i,0])
    
x_test , y_test = np.array(x_test) , np.array(y_test)

y_predicted = model.predict(x_test)

scale_factor = 1/ scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


st.subheader('Preiction vs Orignal || Accuracy graph')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label = 'Original price')
plt.plot(y_predicted,'r',label = 'Predicted price')

plt.xlabel('time')
plt.ylabel('price')
# plt.legend()
plotly_fig = tls.mpl_to_plotly(fig2)
plotly_fig.data[1].line.color = "red"
plotly_fig.data[0].line.color = "blue"
st.plotly_chart(plotly_fig, width=1000, height=1000)
