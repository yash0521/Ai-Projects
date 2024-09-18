import pandas as pd
import numpy as np
import warnings
from glob import glob
import os
from tqdm import tqdm
import pickle
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings('ignore')


def process_raw_csv_data(raw_data):
    """Process raw data and return df"""

    columns_name = [
        "record",
        "time",
        "ax",
        "ay",
        "az",
        "wx",
        "wy",
        "wz",
        "angleX",
        "angleY",
        "angleZ",
        "temp",
        "Unnamed: 13",
    ]

    processed_df = pd.DataFrame([x.split(",") for x in raw_data.split("\n")[1:]],
                                columns=columns_name)

    processed_df = processed_df[["ax", "ay", "az",
                                 "wx", "wy", "wz", "angleX", "angleY", "angleZ"]]

    processed_df.dropna(inplace=True)

    processed_df.reset_index(drop=True, inplace=True)

    return processed_df


# Read csv file from folder
def load_test_dataset(folder_path):
    """
    Load the data from the folder path and return a dataframe
    """
    # Get all the csv files in the folder
    csv_files = glob(os.path.join(folder_path, "*.csv"))

    dataframes = []

    for csv_file in tqdm(csv_files, desc="Loading Data", unit="file"):
        temp_df = pd.read_csv(csv_file, on_bad_lines='skip',
                              na_values='?')

        # Add the dataframe to the list
        dataframes.append(temp_df)

    # Concatenate all the dataframes
    dataset = pd.concat(dataframes)

    column_names = ["ax", "ay", "az", "wx", "wy",
                    "wz", "AngleX", "AngleY", "AngleZ"]
    dataset.columns = column_names

    dataset.dropna(inplace=True)
    dataset.drop_duplicates(inplace=True)
    # dataset = dataset.apply(pd.to_numeric, errors="coerce")

    return dataset

# Read single csv file


def read_file(data_path):
    """
    Load the data from the folder path and return a dataframe
    """

    df = pd.read_csv(data_path, on_bad_lines='skip')
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    return df


def test_data_preprocessing(features):
    """
    This function is used to preprocess the data
    :param features: features
    :return: preprocessed features_scaled
    """

    # Convert the features and labels to numpy array
    standard_scaler = pickle.load(open('utils/StandardScaler.pkl', 'rb'))
    features_scaled = standard_scaler.transform(features)

    return features_scaled


def reshape_features(features):
    """
    This function is used to reshape the data
    :param features: features
    :return: reshaped features
    """

    # Reshape the features for the LSTM
    # features_reshaped = features.reshape(
    #     features.shape[0], 1, features.shape[1])

    # Reshape/Slicing the features for the CNN1d
    window_size = 128
    stride = 64

    sequences = []
    labels = []
    for i in range(len(features) - window_size):
        window = features[i:i+window_size]
        if len(window) == window_size:
            sequences.append(window)

    features_reshaped = np.array(sequences)

    return features_reshaped


def make_prediction(model, features_reshaped):
    """
    Make a prediction with the given model and input.
    :param model: The model to use for prediction.
    :param features_reshaped: The input to the model.
    :return: pred_action: The predicted action.
    """

    # Make a prediction
    pred_action = model.predict(features_reshaped)

    # threshold = 0.5
    # pred_action = np.where(pred_action > threshold, pred_action, 0)

    # Get the index of the class with the maximum count
    pred_action = np.argmax(pred_action, axis=1)

    # Load the label encoder
    label_encoder = pickle.load(open('utils/label_encoder.pkl', 'rb'))

    # # consider the prediction to be the most common label
    prediction = np.bincount(pred_action).argmax()
    most_occurred_pred = label_encoder.classes_[prediction]
    # # print("pred_action: ", prediction)

    # All the actions
    all_actions = label_encoder.inverse_transform(pred_action)

    # Return the prediction
    return all_actions, most_occurred_pred


def ensemble_voting(models, features_reshaped):
    """
    Perform ensemble voting on multiple models and return the most common prediction for each row.
    :param models: A list of CNN models for voting.
    :param features_reshaped: The input to the models.
    :return: all_actions: List of the most common prediction for each row across all models.
             most_occurred_pred: The most common prediction across all models.
    """

    all_predictions = [model.predict(features_reshaped) for model in models]

    # Get the index of the class with the maximum count for each model
    pred_indices = [np.argmax(pred, axis=1) for pred in all_predictions]

    # Combine predictions from all models
    combined_predictions = np.vstack(pred_indices).T

    # Perform voting to get the most common prediction for each row
    ensemble_prediction_per_row = np.apply_along_axis(
        lambda x: np.bincount(x).argmax(), axis=1, arr=combined_predictions)

    # Perform voting to get the most common prediction across all rows
    ensemble_prediction_all_rows = np.bincount(
        ensemble_prediction_per_row).argmax()

    # Load the label encoder
    label_encoder = pickle.load(open('utils/label_encoder.pkl', 'rb'))

    # Get the most common prediction for each row
    most_occurred_pred = label_encoder.classes_[ensemble_prediction_all_rows]

    # Get the most common prediction for each row
    all_actions = [label_encoder.classes_[row_pred]
                   for row_pred in ensemble_prediction_per_row]

    # Return the most common predictions for each row and across all rows
    return all_actions, most_occurred_pred
