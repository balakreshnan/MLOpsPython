# Import libraries
import argparse
from azureml.core import Run
import joblib
import json
import os
import pandas as pd

# Import functions from train.py
from train import split_data, train_model, get_model_metrics

# Get the output folder for the model from the '--output_folder' parameter
parser = argparse.ArgumentParser()
parser.add_argument('--output_folder', type=str, dest='output_folder', default="outputs")
args = parser.parse_args()
output_folder = args.output_folder

# Get the experiment run context
run = Run.get_context()

# load the safe driver prediction dataset
train_df = pd.read_csv('porto_seguro_safe_driver_prediction_input.csv')

# Load the parameters for training the model from the file
with open("parameters.json") as f:
    pars = json.load(f)
    parameters = pars["training"]

# Log each of the parameters to the run
for param_name, param_value in parameters.items():
    run.log(param_name, param_value)
    
# Use the functions imported from train.py to prepare data, train the model, and calculate the metrics
## TODO
data = split_data(train_df)

model = train_model(data, parameters)
predictions = get_model_metrics(model,data)

run.log('Accuracy', predictions)
run.log('auc', predictions['auc'])
run.log('f1score', predictions['f1score'])

# Save the trained model to the output folder
os.makedirs(output_folder, exist_ok=True)
output_path = output_folder + "/driver_model.pkl"
joblib.dump(value=model, filename=output_path)

run.complete()
