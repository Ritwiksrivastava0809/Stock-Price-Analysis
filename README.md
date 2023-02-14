# Stock Price Prediction using Time Series Data and LSTM
This is a machine learning project that uses long short-term memory (LSTM) to predict stock prices based on historical data. It also includes a web application built using Streamlit, allowing users to input a company's stock symbol and view predicted prices.

## Project Overview
The project is structured into the following folders:

Data: contains the historical stock price data in CSV format

Models: contains the trained LSTM model and the scaler used for data normalization

Notebooks: contains Jupyter notebooks used for data exploration, model training and prediction

Webapp: contains the code for the Streamlit web application

## Requirements

To run the project, you need to have Python 3.7 or later installed. The following Python packages are also required:

    pandas
    numpy
    matplotlib
    scikit-learn
    tensorflow
    streamlit
    
To install the required packages, run the following command:
    
    pip install -r requirements.txt
    
    
 ## Usage
 
 ### Model Training and Prediction

To train the LSTM model and generate predictions, run the lstm_prediction.ipynb notebook in the notebooks folder.
This notebook loads the historical stock price data from the data folder, trains the LSTM model, and generates predictions for a specified company's stock.

## Web Application

To run the Streamlit web application, navigate to the webapp folder and run the following command: 

    streamlit run app.py


This will launch the web application in your default web browser. Enter a company's stock symbol and the number of days you would like to predict, and the web app will display a line chart of the predicted stock prices.

## Acknowledgements

This project was inspired by the work of Hvass Laboratories. The stock price data was obtained from Yahoo Finance.


 

![image](https://user-images.githubusercontent.com/112408652/218652372-11ab149a-7361-463b-b56a-f4d09144a317.png)

![image](https://user-images.githubusercontent.com/112408652/218652426-93bf96a9-69e9-446f-8878-47f667e76c41.png)


![image](https://user-images.githubusercontent.com/112408652/218652447-b08bbf48-bc59-49f2-9065-0833a5f1dd41.png)

![image](https://user-images.githubusercontent.com/112408652/218652483-1aff8e20-10d3-4fdb-9164-652474143d7e.png)


![image](https://user-images.githubusercontent.com/112408652/218652507-5d22d623-d46e-4709-bd87-fb92d6762ba9.png)


![image](https://user-images.githubusercontent.com/112408652/218652549-f5a918ff-1e05-4633-b4f1-e2d3c172abb0.png)


![image](https://user-images.githubusercontent.com/112408652/218652571-a030c008-2924-489a-a0d9-4367c6c58d90.png)


