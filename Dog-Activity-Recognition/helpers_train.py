import pandas as pd
import numpy as np
import warnings
from glob import glob
import os
from tqdm import tqdm
import pickle
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

warnings.filterwarnings('ignore')


# '''For Manual Modified data '''
def load_train_dataset(folder_path):
    """
    Load the data from the folder path and return a dataframe
    """
    # Get all the csv files in the folder
    csv_files = glob(os.path.join(folder_path, "*.csv"))

    dataframes = []

    for csv_file in tqdm(csv_files, desc="Loading Data", unit="file"):
        temp_df = pd.read_csv(csv_file, on_bad_lines='skip',
                              na_values='?', header=None)

        # Skip the first 1 row
        temp_df = temp_df.iloc[1:]

        # Add the dataframe to the list
        dataframes.append(temp_df)

    # Concatenate all the dataframes
    dataset = pd.concat(dataframes)

    column_names = ["ax", "ay", "az", "wx", "wy",
                    "wz", "angleX", "angleY", "angleZ", "label"]
    dataset.columns = column_names

    dataset.dropna(inplace=True)
    dataset.drop_duplicates(inplace=True)

    # Converting datatype to float
    columns_to_convert = dataset.columns.difference(['label'])
    dataset[columns_to_convert] = dataset[columns_to_convert].apply(
        pd.to_numeric, errors='coerce')

    dataset.round(2)

    return dataset


# def load_train_data(folder_path):
#     """
#     Load the data from the folder path and return a dataframe
#     """
#     # Get all the csv files in the folder
#     csv_files = glob(os.path.join(folder_path, "*.csv"))

#     dataframes = []

#     for csv_file in tqdm(csv_files, desc="Loading Data", unit="file"):
#         temp_df = pd.read_csv(csv_file, on_bad_lines='skip',
#                               na_values='?')

#          # Get the file name
#         file_name = os.path.basename(csv_file)
#         print(file_name)

#         action = file_name.split(".")[0]
#         print(action)

#         # Add the action as a column
#         temp_df["labels"] = action

#         # Add the dataframe to the list
#         dataframes.append(temp_df)

#     # Concatenate all the dataframes
#     dataset = pd.concat(dataframes)

#     column_names = ["ax", "ay", "az", "wx", "wy",
#                     "wz", "angleX", "angleY", "angleZ", "labels",]
#     dataset.columns = column_names

#     return dataset


def train_data_preprocessing(features, labels):
    """
    This function is used to preprocess the data
    :param features: features
    :param labels: labels
    :return: preprocessed features_scaled, y_categorical
    """
    # Convert the features and labels to numpy array
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Encoding labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_of_classes = len(label_encoder.classes_)

    # One-hot encode the labels
    y_categorical = to_categorical(encoded_labels, num_classes=num_of_classes)

    # Save the label encoder and scaler for later use
    pickle.dump(scaler, open('utils\StandardScaler.pkl', 'wb'))
    pickle.dump(label_encoder, open('utils\label_encoder.pkl', 'wb'))

    return features_scaled, y_categorical
