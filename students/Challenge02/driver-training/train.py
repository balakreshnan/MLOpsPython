import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import lightgbm
from sklearn.metrics import f1_score
# from sklearn.metrics import *


def split_data(data_df):
    """Split a dataframe into training and validation datasets"""

# TODO
    features = data_df.drop(['target', 'id'], axis=1)
    labels = np.array(data_df['target'])
    features_train, features_valid, labels_train, labels_valid = train_test_split(features, labels, test_size=0.2, random_state=0)  # NOQA: E501

    train_data = lightgbm.Dataset(features_train, label=labels_train)
    valid_data = lightgbm.Dataset(features_valid, label=labels_valid, free_raw_data=False)  # NOQA: E501

    return (train_data, valid_data)


def train_model(data, parameters):
    """Train a model with the given datasets and parameters"""
    # The object returned by split_data is a tuple.
    # Access train_data with data[0] and valid_data with data[1]

# TODO
    model = lightgbm.train(parameters,
                           data[0],
                           valid_sets=data[1],
                           num_boost_round=500,
                           early_stopping_rounds=20)

    return model


def get_model_metrics(model, data):
    """Construct a dictionary of metrics for the model"""

# TODO
    predictions = model.predict(data[1].data)
    fpr, tpr, thresholds = metrics.roc_curve(data[1].label, predictions)
    # lgb_prediction = predictions.argmax(axis=0)
    lgb_F1 = f1_score(data[1].label, data[1].label, average='weighted')
    # roc_auc_score(lgb_prediction, data[1].data)
    # recall_score(lgb_prediction, data[1].data)
    model_metrics = {"auc": (metrics.auc(fpr, tpr)), "f1score": lgb_F1}
    print(model_metrics)

    return model_metrics


def main():
    """This method invokes the training functions for development purposes"""

    # Read data from a file
    data_df = pd.read_csv('porto_seguro_safe_driver_prediction_input.csv')

    # Hard code the parameters for training the model
    parameters = {
        'learning_rate': 0.02,
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'sub_feature': 0.7,
        'num_leaves': 60,
        'min_data': 100,
        'min_hessian': 1,
        'verbose': 2
    }

    # Call the functions defined in this file
    data = split_data(data_df)
    model = train_model(data, parameters)
    predictions = get_model_metrics(model, data)

    # Print the resulting metrics for the model
# TODO
    print(predictions)


if __name__ == '__main__':
    main()
