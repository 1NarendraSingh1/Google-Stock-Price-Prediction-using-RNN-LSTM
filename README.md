# Google-Stock-Price-Prediction-using-RNN-LSTM
## 🛠️ Tools and Libraries  
- **Libraries Used:** numpy, pandas, matplotlib, sklearn.preprocessing, keras (for deep learning models)
- **Dataset:** Google_train_data.csv,Google_test_data.csv, containing columns like Date, Open, High, Low, Close, and Volume
- **Deep Learning Approach:** LSTM (Long Short-Term Memory), Dense, LSTM, Dropout, Adam Optimizer
- **Data Preprocessing:** MinMaxScaler is used for normalization
- **Visualization:** matplotlib

# Stock Price Prediction Using Deep Learning
**Overview**
- This project focuses on predicting stock prices using deep learning techniques, specifically Long Short-Term Memory (LSTM) networks. The dataset consists of historical stock prices, and the model aims to forecast future prices based on past trends.

**Libraries Used**
The following libraries are utilized in this project:

- NumPy: For numerical computations and array manipulation.
- Pandas: Used for data manipulation, reading datasets, and handling time-series data.
- Matplotlib: A visualization library for plotting stock trends and model predictions.
- sklearn.preprocessing: Provides MinMaxScaler for normalizing stock prices, ensuring values are within a specified range for better model performance.
- Keras: A deep learning library used to build and train the LSTM model.

**Dataset**
The dataset consists of historical stock price data from Google_train_data.csv and Google_test_data.csv, containing the following columns:

- Date: The trading date.
- Open: The stock’s opening price.
- High: The highest price of the stock on that day.
- Low: The lowest price of the stock on that day.
- Close: The closing price of the stock.
- Volume: The total number of shares traded on that day.

  
**Deep Learning Approach**
The stock price prediction model is built using the Long Short-Term Memory (LSTM) architecture, which is a type of recurrent neural network (RNN) designed to handle sequential data.

The model consists of the following layers:

- LSTM Layer: Captures long-term dependencies in time-series data.
- Dense Layer: Fully connected layer to generate predictions.
- Dropout Layer: Prevents overfitting by randomly deactivating neurons during training.
- Adam Optimizer: An adaptive learning rate optimization algorithm used for efficient training.

**Data Preprocessing**
Before training, the data undergoes preprocessing to enhance model accuracy:

- MinMaxScaler (from sklearn.preprocessing) is used to normalize stock prices between 0 and 1, ensuring stable and faster training.
- Data is reshaped into a format suitable for LSTM models.
- A rolling window approach is applied to create sequences of past stock prices as input for the model.

**Visualization**
The matplotlib library is used for various visualizations, including:

- Stock Price Trends: Line charts displaying historical price movements.
- Training vs. Predicted Prices: Comparison of actual vs. predicted values to evaluate model performance.

pip install numpy pandas matplotlib scikit-learn keras tensorflow
Load and preprocess the dataset using pandas and MinMaxScaler.
Build and train the LSTM model using Keras.
Visualize the results using matplotlib.

**Conclusion**

This project demonstrates the use of deep learning (LSTM) for stock price prediction. By leveraging historical stock price data, we can generate predictions that may assist in decision-making for traders and analysts.
