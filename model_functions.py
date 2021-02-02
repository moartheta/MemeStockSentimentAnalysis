#Imports
import pandas as pd
import numpy as np
import pandas as pd
import plotly.express as px
import hvplot.pandas

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

#Defining function for the three-feature model
def get_mse(ticker, num_units, dropout, epochs, batch_size, window_size, train_size, target_col_number):

    df = pd.read_csv(f'Data/Master/{ticker}.csv', index_col="Created", infer_datetime_format=True, parse_dates=True)
    df = df.dropna()

    #Getting window data
    X = []
    y = []
    for i in range(len(df) - window_size - 1):
        features = df.iloc[i:(i + window_size), :]
        target = df.iloc[(i + window_size), target_col_number]
        X.append(features)
        y.append(target)

    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    
    #Splitting the data
    split = int(train_size * len(X))
    X_train = X[: split -1]
    X_test = X[split:]
    y_train = y[: split -1]
    y_test = y[split:]
    
    #Scaling the data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    scaler.fit(y_train)
    y_train = scaler.transform(y_train)
    y_test = scaler.transform(y_test)
    
    #Reshaping the data
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 3))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1],3))
    
    #Initializing the model
    model = Sequential()
    number_units = num_units
    dropout_fraction = dropout
    
    #Layer1
    model.add(LSTM(
    units = number_units,
    return_sequences = True,
    input_shape = (X_train.shape[1], 3)))
    model.add(Dropout(dropout_fraction))
    #Layer2
    model.add(LSTM(units = number_units, return_sequences = True))
    model.add(Dropout(dropout_fraction))
    #Layer3
    model.add(LSTM(units=number_units))
    model.add(Dropout(dropout_fraction))
    #Output
    model.add(Dense(1))

    #Compiling the model
    model.compile(optimizer = "adam", 
                  loss = "mean_squared_error",
                     )

    #Fitting the model
    model.fit(
            X_train, y_train,
            epochs = epochs,
            shuffle = False,
            batch_size = batch_size, verbose = 1)

    #Making predictions
    predicted = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted)
    real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

    stocks = pd.DataFrame({
                "Real": real_prices.ravel(),
                "Predicted": predicted_prices.ravel()
                }, index = df.index[-len(real_prices): ]) 

    error = model.evaluate(X_test, y_test)
    
    MSE = []

    MSE.append(ticker)
    MSE.append(error)
        
    return stocks, MSE

#Defining function for one-feature model
def get_one_feature_model(ticker, num_units, dropout, epochs, batch_size, window_size, train_size, target_col_number, feature_col_number):
   
    df = pd.read_csv(f'Data/Master/{ticker}.csv', index_col="Created", infer_datetime_format=True, parse_dates=True)
    df = df.dropna()

    #Getting window data
    X = []
    y = []
    for i in range(len(df) - window_size - 1):
        features = df.iloc[i:(i + window_size), feature_col_number]
        target = df.iloc[(i + window_size), target_col_number]
        X.append(features)
        y.append(target)

    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    
    #Splitting the data
    split = int(train_size * len(X))
    X_train = X[: split -1]
    X_test = X[split:]
    y_train = y[: split -1]
    y_test = y[split:]
    
    #Scaling the data
    scaler = MinMaxScaler()
    
    scaler.fit(X)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    scaler.fit(y)
    y_train = scaler.transform(y_train)
    y_test = scaler.transform(y_test)
    
    #Reshaping the data
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    #Initializing the model
    model = Sequential()
    number_units = num_units
    dropout_fraction = dropout
    
    #Layer1
    model.add(LSTM(
    units = number_units,
    return_sequences = True,
    input_shape = (X_train.shape[1], 1)))
    model.add(Dropout(dropout_fraction))
    #Layer2
    model.add(LSTM(units = number_units, return_sequences = True))
    model.add(Dropout(dropout_fraction))
    #Layer3
    model.add(LSTM(units=number_units))
    model.add(Dropout(dropout_fraction))
    #Output
    model.add(Dense(1))

    #Compiling the model
    model.compile(optimizer = "adam", loss = "mean_squared_error")

    #Fitting the model
    model.fit(
            X_train, y_train,
            epochs = epochs,
            shuffle = False,
            batch_size = batch_size, verbose = 1)

    #Making predictions
    predicted = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted)
    real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

    stocks = pd.DataFrame({
                "Real": real_prices.ravel(),
                "Predicted": predicted_prices.ravel()
                }, index = df.index[-len(real_prices): ]) 

    error = model.evaluate(X_test, y_test)

    MSE = []

    MSE.append(ticker)
    MSE.append(error)
        
    return stocks, MSE


#Defining function to get twitter sentiment joined together and saved to CSVs
def twitter_sentiment_scores(tickers):
    
    for symbol in tickers:
        
        df = pd.read_csv(f'Data/{symbol}_tweets.csv', index_col="Created", infer_datetime_format=True, parse_dates=True)
        df2 = pd.read_csv(f'Data/tweets_{symbol}_sentiment.csv', index_col="Created", infer_datetime_format=True, parse_dates=True)

        df['Sentiment'] = df['Sentiment'].str.replace('{','')
        df['Sentiment'] = df['Sentiment'].str.replace('}','')
        df['Sentiment'] = df['Sentiment'].str.replace("'",'')
        df['Sentiment'] = df['Sentiment'].str.replace("basic",'')
        df['Sentiment'] = df['Sentiment'].str.replace(": ",'')

        df.index = pd.to_datetime(df.index, utc = True)
        df.index = df.index.date
        df.sort_index(inplace=True)
        df.dropna(inplace=True)

        df['Sentiment'] = np.where(df["Sentiment"] == 'Bullish', 1, -1)

        df['Like Score'] = df["Likes"] * df["Sentiment"]

        df_gb = df.groupby(df.index).mean()

        joined_sent = df_gb.merge(df2, left_index=True, right_index=True)
        
        joined_sent['NLTK_Compound'] = joined_sent['NLTK_Compound'] * (joined_sent['Like Score'].mean())
        joined_sent['Blob Score'] = joined_sent['Blob Score'] * (joined_sent['Like Score'].mean())
        
        filename = 'Data/MasterSentiment/master_sentiment_{}.csv'.format(symbol)
        joined_sent.to_csv(filename, header=True, index=True)
        
        print(f"{symbol} saved to csv!")
        
#Defining a function that gathers the current data for price vs settlement and saves as CSVs
        def price_vs_sentiment(tickers):
    
    for symbol in tickers:
        
        tweets = pd.read_csv(f'Data/reddit_{symbol}_sentiment.csv', index_col="Created", infer_datetime_format=True, parse_dates=True)
        prices = pd.read_csv('Data/alpaca_data.csv', index_col="Created", infer_datetime_format=True, parse_dates=True).sort_index()
        
        tweets.index = pd.to_datetime(tweets.index, utc = True)
        tweets.index = tweets.index.date
        
        prices = prices[[symbol.upper()]]
        prices.index = pd.to_datetime(prices.index, utc = True)
        prices.index = prices.index.date
        
        combined = tweets.merge(prices, left_index=True, right_index=True).dropna()
        
        #combined['NLTK_Compound'] = (combined['NLTK_Compound']) * (combined[symbol.upper()].mean())
        #combined['Blob Score'] = (combined['Blob Score']) * (combined[symbol.upper()].mean())
        
        filename = 'Data/MasterSentiment/VersusPrice/price_vs_sentiment_{}.csv'.format(symbol)
        combined.to_csv(filename, header=True, index=True)
        
        print(f"{symbol} saved to csv!")