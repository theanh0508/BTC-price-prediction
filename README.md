# Bitcoin Price Prediction Project

## Overview

This repository contains code and resources for predicting Bitcoin prices using time-series analysis and deep learning techniques. The project involves downloading updated BTC price data from the Coinranking website using Python API requests, performing data exploratory and data cleaning, creating an interactive dashboard to display monthly high and low BTC prices from 2019 to 2022, and benchmarking BTC price prediction using various deep learning algorithms. 

## Project Highlights

- **Data Collection:**
  - Utilizing Python API requests to download up-to-date BTC price data from the Coinranking website.

- **Data Exploration and Cleaning:**
  - Conducting exploratory data analysis to gain insights into the dataset.
  - Implementing data cleaning processes to ensure data accuracy and reliability.

- **Interactive Dashboard:**
  - Developing an interactive dashboard to visualize monthly high and low BTC prices from 2019 to 2022.

- **Baseline Performance:**
  - Conducting a Naive Forecast and Moving Average as baseline performance metrics.

- **Time Series Analysis:**
  - Implementing seasonal decomposition of time series to detect seasonal trends in Bitcoin prices.

- **Deep Learning Models:**
  - Building and training a deep neural network using a time window of 30 data points to predict the next point, achieving an MAE of 516.
  - Building and training a Recurrent Neural Network (RNN), obtaining an MAE of 466.
  - Building and training a Long Short-Term Memory (LSTM) model, achieving an MAE 473.

- Please look at the Jupyter notebook file for further analysis and conclusions 
