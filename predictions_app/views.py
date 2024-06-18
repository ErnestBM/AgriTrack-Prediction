from django.http import JsonResponse
import os
import pandas as pd
import numpy as np
import json
import requests
from sklearn.preprocessing import MinMaxScaler

from .utils import replace_comma, convert_to_numeric, handle_missing_values, normalize_data, one_hot_encode_with_unique_values, split_data, create_model, download_weights, predict_future_with_simulation

def predict_commodity(request, commodity):
    # Assuming your model and data preprocessing utilities are in a separate file called utils.py
    # Make sure to import all necessary functions from utils.py

    # Path to the CSV data for the commodity
    data_path = f'data/{commodity}.csv'

    # Check if the data file exists
    if not os.path.exists(data_path):
        return JsonResponse({'error': f'Data file for {commodity} not found.'}, status=404)

    # Load the CSV data into a DataFrame
    df = pd.read_csv(data_path)

    # List of columns that need to be preprocessed
    columns_to_convert = ['Tn', 'Tx', 'Tavg', 'RR', 'ss']

    # Preprocess the data
    df = replace_comma(df, columns_to_convert)
    df = convert_to_numeric(df, columns_to_convert)
    df = handle_missing_values(df)

    # Columns to exclude from normalization
    exclude_columns = [commodity, 'Date', 'ddd_car']

    # Normalize the data
    df, feature_scaler, target_scaler = normalize_data(df, exclude_columns)

    # Unique values for one-hot encoding
    unique_values = {'SE', 'E', 'N', 'W', 'NW', 'SW', 'NE', 'S', 'C'}
    df = one_hot_encode_with_unique_values(df, unique_values, 'ddd_car')

    # Split data into train, validation, and test sets
    train, val, test = split_data(df)

    # Get the features for prediction
    features = df.drop(columns=[commodity, "Date"]).columns.tolist()

    # Prepare input data for prediction
    X_test = test[features].values

    # Scale the input data
    scaler = MinMaxScaler()
    X_test = scaler.fit_transform(X_test)

    # Define the model parameters
    timestep = 30
    feature_count = len(features)

    # URL to download model weights
    url = f'https://storage.googleapis.com/agritrack-prediction-bucket/{commodity.lower().replace(" ", "_")}.weights.h5'

    # File path to save the model weights
    weights_file_path = f'{commodity.lower().replace(" ", "_")}.weights.h5'

    # Download the model weights if not already downloaded
    download_weights(url, weights_file_path)

    # Create the model
    model = create_model(timestep, feature_count)

    # Load the model weights
    model.load_weights(weights_file_path)

    # Prepare initial input data for prediction
    initial_input_data = X_test[-timestep:]

    # Number of future steps to predict
    future_steps = 30

    # Last known features for prediction
    last_known_features = X_test[-1]

    # Predict future values
    future_predictions = predict_future_with_simulation(model, initial_input_data, future_steps, timestep,
                                                       feature_count, target_scaler, scaler, last_known_features)

    # Format predictions into JSON response
    predictions = {
        'commodity': commodity,
        'predictions': [float(pred) for pred in future_predictions]
    }

    return JsonResponse(predictions)