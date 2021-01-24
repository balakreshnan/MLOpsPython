# Import libraries
from azureml.core import Run
import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
import lightgbm
from sklearn import metrics

# Get parameters
parser = argparse.ArgumentParser()
parser.add_argument('--output_folder', type=str, dest='output_folder', default="diabetes_model", help='output folder')
args = parser.parse_args()
output_folder = args.output_folder

from azureml.core import Workspace
from azureml.core import Dataset
ws = Workspace.get(name='mlopsdev',
           subscription_id='c46a9435-c957-4e6c-a0f4-b9a597984773',
           resource_group='mlops'
)

#ws = Workspace.from_config()

# Get the experiment run context
run = Run.get_context()

# load the safe driver prediction dataset
#train_df = pd.read_csv('porto_seguro_safe_driver_prediction_input.csv')
#train_df = run.input_datasets['driversdataset'].to_pandas_dataframe()
dataset = Dataset.get_by_name(ws, name='driversdataset')
data_df = dataset.to_pandas_dataframe()

# Load the parameters for training the model from the file
#with open("parameters.json") as f:
#    pars = json.load(f)
#    parameters = pars["training"]
    
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

# Log each of the parameters to the run
for param_name, param_value in parameters.items():
    run.log(param_name, param_value)

features = data_df.drop(['target', 'id'], axis=1)
labels = np.array(data_df['target'])
(features_train, features_valid, labels_train, labels_valid) = train_test_split(features, labels, test_size=0.2, random_state=0)

train_data = lightgbm.Dataset(features_train, label=labels_train)
valid_data = lightgbm.Dataset(features_valid, label=labels_valid, free_raw_data=False)
    
model = lightgbm.train(parameters, train_data, valid_sets=valid_data, num_boost_round=500, early_stopping_rounds=20)
    
#model = train_model(train_data, valid_data, parameters)
#predictions = get_model_metrics(model, valid_data)

predictions = model.predict(valid_data.data)
fpr, tpr, thresholds = metrics.roc_curve(valid_data.label, predictions)
model_metrics = {"auc": (metrics.auc(fpr, tpr))}
print(model_metrics)

run.log('Accuracy', model_metrics)
run.log('ModelType', 'LightGbm')

# Save the trained model to the output folder
os.makedirs(output_folder, exist_ok=True)
output_path = output_folder + "/driver_model.pkl"
joblib.dump(value=model, filename=output_path)

print(output_path)
print(model)

# Save the trained model
#os.makedirs(output_folder, exist_ok=True)
#output_path = output_folder + "/model.pkl"
#joblib.dump(value=model, filename=output_path)

run.complete()
