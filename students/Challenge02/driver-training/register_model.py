# Import libraries
import argparse
import joblib
from azureml.core import Run

# Get parameters
parser = argparse.ArgumentParser()
parser.add_argument('--model_folder', type=str, dest='model_folder', default="driver_model", help='model location')  # NOQA: E501
args = parser.parse_args()
model_folder = args.model_folder

# Get the experiment run context
run = Run.get_context()

# load the model
print("Loading model from " + model_folder)
model_name = 'driver_model'
model_file = model_folder + "/" + model_name + ".pkl"

metrics = run.get_metrics(recursive=True)

# Load the model
print("Loading model from " + model_folder)
model_file = model_folder + "/driver_model.pkl"
model = joblib.load(model_file)

# run.upload_file('driver_model.pkl',model_file)
run.upload_file(model_name, model_file)

# run.register_model(model_path = model_file,
#                   model_name = 'driver_model.pkl',
#                   tags=metrics)
run.register_model(model_path=model_name,
                   model_name=model_name,
                   tags=metrics)

run.complete()
