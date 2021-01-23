import pandas as pd
import numpy as np
import pandas as pd
import plotly.express as px
#pn.extension('plotly')
import hvplot.pandas

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class LST:
    """
    Initialize, then run 'predict'
    Initializing will set the following default values:
      dataframe (passed in)
      y_col =0
      num_units = 10
      dropout = 0.2
      epochs = 15
      batch_size = 2
      window_size = 2
      train_size = 0.7
      
    predict requires feature_col (int) and
    target_col (int), returns dataframe
    
    """
    
    def __init__(self, df = None):
        #set default values
        
        self.df = df
        self.y_col = 0
        self.num_units = 10
        self.dropout = 0.2
        self.epochs = 15
        self.batch_size = 2
        self.window_size = 2
        self.train_size = 0.7
        #--------------------------------------
        
    def window_data(self,feature_col_number, target_col_number):
        X = []
        y = []
        for i in range(len(self.df) - self.window_size - 1):
            features = self.df.iloc[i:(i + self.window_size), feature_col_number]
            target = self.df.iloc[(i + self.window_size), target_col_number]
            X.append(features)
            y.append(target)
        return np.array(X), np.array(y).reshape(-1, 1)
    
    def make_dfs(self,df_ticker, df_features):
        """
        Will return a list of dataframes based on number of columns (tickers)
        Sandbox for now, but we can adjust as needed to re-configure dfs
        
        """
        
        dfs = []
        counter = 0
        for ticker in range(counter,df_ticker.columns.size):
            df = pd.DataFrame({
                df_ticker.columns.values[counter] : df_ticker[df_ticker.columns.values[counter]]
            })
            df = pd.concat([df, df_features], axis = 1, sort=False)
            dfs.append(df)
            counter += 1        
        return dfs
        
    #setter functions
    #used to override the default values from the constructor
    def set_num_units(self, val):
        self.num_units = val
    def set_dropout(self, val):
        self.dropout = val
    def set_epochs(self, val):
        self.epochs = val
    def set_batch_size(self):
        self.batch_size = val
    def set_window_size(self, val):
        self.window_size = val
    def set_train_size(self,val):
        self.train_size = val
        
    #should be working correctly, but having issues when calling from predict method    
    def split_me(self, X, y):
        """
        Splits data into training variables, returns all 4 so must be unpacked by 4 when called
        """
        split = int(self.train_size * len(X))
        X_train = X[: split -1]
        X_test = X[split:]
        y_train = y[: split -1]
        y_test = y[split:]
        return X_train, X_test, y_train, y_test
    
    #should be working correctly, but having issues when calling from predict method
    #specifically the scaler, and the shape
    def scale_me(self, X_train, X_test, y_train, y_test):
        """
        Must pass 4 training variables: X_train, X_test, y_train, and y_test
        Returns 4 scaled training variables, X_train and X_test are reshaped.
        Must be unpacked by 4 variables when called
        """
        scaler = MinMaxScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        scaler.fit(y_train)
        y_train = scaler.transform(y_train)
        y_test = scaler.transform(y_test)

        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1],1))
        return X_train,X_test,y_train,y_test
        
    def predict(self,feature_col_num, target_col_num):
        """
        Dataframe must be prepared prior to passing in
        will return prediction dataframe
        """
        
        X, y = self.window_data(feature_col_num, target_col_num)

        X_train, X_test, y_train, y_test = self.split_me(X, y)        
        
        X_train, X_test, y_train, y_test = self.scale_me(X_train, X_test, y_train, y_test)
        
        
        #Might be tricky to separate this out right now due to coding for layers
        model = Sequential()
        number_units = self.num_units
        dropout_fraction = self.dropout

        #Layer1
        model.add(LSTM(
        units = number_units,
        return_sequences = True,
        input_shape = (X_train.shape[1], 1))
        )

        model.add(Dropout(dropout_fraction))
        #Layer2
        model.add(LSTM(units = number_units, return_sequences = True))
        model.add(Dropout(dropout_fraction))
        #Layer3
        model.add(LSTM(units=number_units))
        model.add(Dropout(dropout_fraction))
        #Output
        model.add(Dense(1))

        model.compile(optimizer = "adam", loss = "mean_squared_error")

        model.fit(
            X_train, y_train,
            epochs = self.epochs,
            shuffle = False,
            batch_size = self.batch_size, verbose = 1
        )

        predicted = model.predict(X_test)
        scaler = MinMaxScaler().fit(y_train)
        predicted_prices = scaler.inverse_transform(predicted)
        real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

        stocks = pd.DataFrame({
            "Real": real_prices.ravel(),
            "Predicted": predicted_prices.ravel()
            }, index = self.df.index[-len(real_prices): ]) 

        return predicted_prices
    
        #basic starter plots - expand on this later.  possibly create separate file
        def plot_me(self, df):
            return df.plot()
        
        def hvscatter(self,df,x,y, title = "Scatter Plot"):
            return df.hvplot.scatter(
                x = x,
                y = y,
                title = title
            )
        
        def pxscatter(self,df,x,y,title = "Scatter Plot"):
            return px.scatter(
                df,
                x = x,
                y = y,
                title = title
            )