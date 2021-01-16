class LSTM:
    def __init__(self, df = None):
        #Move these to the top?
        import numpy as np
        import pandas as pd
        import hvplot.pandas
        from sklearn.preprocessing import MinMaxScaler
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        #set default values
        
        self.df = df
        self.y_col = 0
        self.num_units = 10
        self.dropout = 0.2
        self.epochs = 15
        self.batch_size = 2
        self.window_size = 10
        
    def window_data(self,feature_col_number, target_col_number):
        X = []
        y = []
        for i in range(len(self.df) - window - 1):
            features = self.df.iloc[i:(i + window), feature_col_number]
            target = self.df.iloc[(i + window), target_col_number]
            X.append(features)
            y.append(target)
        return np.array(X), np.array(y).reshape(-1, 1)
    
    #setter functions
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
        
    def predict(self,feature_col=None, target_col=None):
        """
        Dataframe must be prepared prior to passing in
        will return prediction dataframe
        """
        X, y = window_data(self.df, window_size, feature_col, target_col)
        
        #Make separate method?
        split = int(0.7 * len(X))
        X_train = X[: split -1]
        X_test = X[split:]
        y_train = y[: split -1]
        y_test = y[split:]
        #----------------------
        
        #Make separate method?
        scaler = MinMaxScaler().fit(X)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        scaler.fit(y)
        y_train = scaler.transform(y_train)
        y_test = scaler.transform(y_test)

        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1],1))
        #-------------------------------------------------------------------
        
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

        predicted_prices = scaler.inverse_transform(predicted)
        real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

        stocks = pd.DataFrame({
            "Real": real_prices.ravel(),
            "Predicted": predicted_prices.ravel()
            }, index = df.index[-len(real_prices): ]) 

        return stocks
    
    
        #expand on this later
        def plot_me(self, df):
            return df.plot()