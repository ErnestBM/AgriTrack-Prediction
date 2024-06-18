import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import json
import requests
import os

def replace_comma(df, columns):
    for col in columns:
        df[col] = df[col].str.replace(',', '.')
    return df

def convert_to_numeric(df, columns):
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def handle_missing_values(df):
    return df.fillna(method='ffill').fillna(method='bfill')

def normalize_data(df, exclude_columns):
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    features = df.drop(columns=exclude_columns)

    scaled_features = feature_scaler.fit_transform(features)
    scaled_target = target_scaler.fit_transform(df[[exclude_columns[0]]])

    scaled_df = pd.DataFrame(scaled_features, columns=features.columns)
    scaled_df[exclude_columns[0]] = scaled_target

    for col in exclude_columns[1:]:
        scaled_df[col] = df[col].values

    return scaled_df, feature_scaler, target_scaler

def one_hot_encode_with_unique_values(df, unique_values, column):
    for value in unique_values:
        df[column + '_' + value] = (df[column] == value).astype(int)
    df.drop(columns=[column], inplace=True)
    return df

def split_data(df, train_ratio=0.8, val_ratio=0.1):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train_data = df[:train_end]
    val_data = df[train_end:val_end]
    test_data = df[val_end:]
    return train_data, val_data, test_data

def create_model(timestep, feature_count):
    model = Sequential()
    model.add(Bidirectional(LSTM(units=200, return_sequences=True), input_shape=(timestep, feature_count)))
    model.add(Dropout(0.4))
    model.add(Bidirectional(GRU(units=150, return_sequences=True)))
    model.add(Dropout(0.4))
    model.add(LSTM(units=75, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(GRU(units=50))
    model.add(Dropout(0.4))
    model.add(Dense(units=1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def download_weights(url, weights_file_path):
    if not os.path.exists(weights_file_path):
        response = requests.get(url)
        with open(weights_file_path, 'wb') as file:
            file.write(response.content)

def predict_future_with_simulation(model, data, steps, timestep, feature_count, target_scaler, feature_scaler,
                                   last_known_features):
    future_predictions = []
    current_input = data[-timestep:]

    for _ in range(steps):
        current_input = current_input.reshape((1, timestep, feature_count))

        next_prediction = model.predict(current_input)
        next_value = next_prediction[0][0]

        future_predictions.append(next_value)

        next_input = current_input[:, -1, :].copy()
        next_input[:, -1] = next_value

        for i in range(feature_count - 1):
            next_input[:, i] = last_known_features[i]

        current_input = np.append(current_input[:, 1:, :], next_input.reshape(1, 1, feature_count), axis=1)

    return future_predictions

def main(commodity):
    df = pd.read_csv(f'data/{commodity}.csv')

    columns_to_convert = ['Tn', 'Tx', 'Tavg', 'RR', 'ss']
    df = replace_comma(df, columns_to_convert)
    df = convert_to_numeric(df, columns_to_convert)
    df = handle_missing_values(df)

    exclude_columns = [commodity, 'Date', 'ddd_car']
    df, feature_scaler, target_scaler = normalize_data(df, exclude_columns)

    unique_values = {'SE', 'E', 'N', 'W', 'NW', 'SW', 'NE', 'S', 'C'}
    df = one_hot_encode_with_unique_values(df, unique_values, 'ddd_car')

    train, val, test = split_data(df)

    features = df.drop(columns=exclude_columns).columns.tolist()
    X_test = test[features].values
    y_test = test[commodity].values

    scaler = MinMaxScaler()
    X_test = scaler.fit_transform(X_test)

    timestep = 30
    feature_count = len(features)

    url = f'https://storage.googleapis.com/agritrack-prediction-bucket/{commodity.lower().replace(" ", "_")}.weights.h5'
    weights_file_path = f'{commodity.lower().replace(" ", "_")}.weights.h5'

    download_weights(url, weights_file_path)

    model = create_model(timestep, feature_count)
    model.load_weights(weights_file_path)

    initial_input_data = X_test[-timestep:]
    future_steps = 30
    last_known_features = X_test[-1]

    future_predictions = predict_future_with_simulation(model, initial_input_data, future_steps, timestep,
                                                       feature_count, target_scaler, scaler, last_known_features)

    future_predictions_output = {
        "future_predictions": [float(pred) for pred in future_predictions]
    }

    with open(f'{commodity.lower().replace(" ", "_")}_pred.json', 'w') as f:
        json.dump(future_predictions_output, f, indent=2)

if __name__ == '__main__':
    commodities = [
        'Bawang Putih Bonggol',
        'Beras Premium',
        'Beras Medium',
        'Daging Ayam Ras',
        'Telur Ayam Ras',
        'Gula Konsumsi',
        'Tepung Terigu',
        'Minyak Goreng Curah',
        'Ikan Tongkol',
        'Garam Halus'
    ]

    for commodity in commodities:
        main(commodity)
