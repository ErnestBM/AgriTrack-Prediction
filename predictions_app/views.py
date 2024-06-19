from django.http import JsonResponse
import os
import pandas as pd
import numpy as np
import json
import requests
from sklearn.preprocessing import MinMaxScaler
import sqlite3


from .utils import replace_comma, convert_to_numeric, handle_missing_values, normalize_data, one_hot_encode_with_unique_values, split_data, create_model, download_weights, predict_future_with_simulation

def predict_commodity(request, commodity):

    data_path = f'data/{commodity}.csv'

    if not os.path.exists(data_path):
        return JsonResponse({'error': f'Data file for {commodity} not found.'}, status=404)

    df = pd.read_csv(data_path)

    columns_to_convert = ['Tn', 'Tx', 'Tavg', 'RR', 'ss']

    df = replace_comma(df, columns_to_convert)
    df = convert_to_numeric(df, columns_to_convert)
    df = handle_missing_values(df)

    exclude_columns = [commodity, 'Date', 'ddd_car']

    df, feature_scaler, target_scaler = normalize_data(df, exclude_columns)

    unique_values = {'SE', 'E', 'N', 'W', 'NW', 'SW', 'NE', 'S', 'C'}
    df = one_hot_encode_with_unique_values(df, unique_values, 'ddd_car')

    train, val, test = split_data(df)

    features = df.drop(columns=[commodity, "Date"]).columns.tolist()

    X_test = test[features].values

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

    predictions = {
        'commodity': commodity,
        'predictions': [float(pred) for pred in future_predictions]
    }

    return JsonResponse(predictions)