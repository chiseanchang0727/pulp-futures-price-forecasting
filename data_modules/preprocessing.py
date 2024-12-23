import re
import pandas as pd


def clean_data(input_df: pd.DataFrame) -> pd.DataFrame:
    df = input_df.copy()
    cols = ['Price', 'Open', 'High', 'Low', 'Vol.', 'Change %']

    for col in cols:
        if col == 'Vol.':
            df[col] = df[col].apply(lambda x: float(re.sub('K', '', x)) * 1000 if x != '-' else 0)
        elif col == 'Change %':
            df[col] = df[col].apply(lambda x: float(re.sub('%', '', x)) / 100)
        else:
            df[col] = df[col].apply(lambda x: float(re.sub(',', '', x)))
    
    return df



def feature_engineering(input_df: pd.DataFrame) -> pd.DataFrame:
    df = input_df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Time-based features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek

    # Lag features (previous days' prices)
    df['Price_Lag_1'] = df['Price'].shift(1)
    df['Price_Lag_3'] = df['Price'].shift(3)
    df['Price_Lag_7'] = df['Price'].shift(7)

    # Rolling features (moving averages and rolling statistics)
    df['Price_RollingMean_3'] = df['Price'].rolling(window=3).mean()
    df['Price_RollingMean_7'] = df['Price'].rolling(window=7).mean()
    df['Price_RollingStd_3'] = df['Price'].rolling(window=3).std()
    df['Price_RollingStd_7'] = df['Price'].rolling(window=7).std()

    # Percentage change features
    df['Price_PctChange'] = df['Price'].pct_change()

    # Features derived from High and Low
    df['High_Low_Spread'] = df['High'] - df['Low'] 
    df['High_Low_Avg'] = (df['High'] + df['Low']) / 2
    df['High_Low_PctChange'] = (df['High'] - df['Low']) / df['Low'] 

    # Interaction features
    df['High_Open_Spread'] = df['High'] - df['Open'] 
    df['Low_Close_Spread'] = df['Low'] - df['Price']  

    # Lag features for High and Low
    df['High_Lag_1'] = df['High'].shift(1)
    df['Low_Lag_1'] = df['Low'].shift(1)
    df['High_Low_Spread_Lag_1'] = df['High_Low_Spread'].shift(1)

    # Rolling statistics for High and Low
    df['High_RollingMean_3'] = df['High'].rolling(window=3).mean()
    df['Low_RollingMean_3'] = df['Low'].rolling(window=3).mean()
    df['High_RollingStd_3'] = df['High'].rolling(window=3).std()
    df['Low_RollingStd_3'] = df['Low'].rolling(window=3).std()

    # Features derived from Volume (Vol.)
    df['Vol_RollingMean_3'] = df['Vol.'].rolling(window=3).mean()  # 3-day moving average of volume
    df['Vol_RollingMean_7'] = df['Vol.'].rolling(window=7).mean()  # 7-day moving average of volume
    df['Vol_RollingStd_3'] = df['Vol.'].rolling(window=3).std()    # 3-day rolling std of volume
    df['Vol_RollingStd_7'] = df['Vol.'].rolling(window=7).std()    # 7-day rolling std of volume
    df['Vol_PctChange'] = df['Vol.'].pct_change()  # Percentage change in volume
    df['Vol_Lag_1'] = df['Vol.'].shift(1)          # Lag feature for volume


    
    # Fill missing values created by lagging and rolling
    df.fillna(0, inplace=True)

    return df



def data_preprocessing(raw_data: pd.DataFrame) -> pd.DataFrame:
    data = feature_engineering(raw_data)
    
    data = clean_data(data)

    data = data.sort_values('Date').reset_index(drop=True)
    
    data = data.drop('Date', axis=1)

    return data
