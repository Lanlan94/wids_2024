import os
import yaml
import joblib
import pandas as pd
import os
import datetime

def load_data(source_file):
    data = pd.read_csv(source_file)
    return data

def create_directory():
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    directory_name = f"run_{timestamp}"

    directory_path = os.path.join('runs', directory_name)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return directory_path

def safe_data_to_directory(data,directory='/',fname='training.csv'):
    path = os.path.join(directory, fname)
    data.to_csv(path)
    try:
        data.to_csv(path, index=False)
        return True
    except Exception as e:
        print(f"Error saving data to {path}: {str(e)}")
        return False 
    
def save_dict_as_yaml(data, file_path):
    with open(file_path, 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)

def load_yaml_as_dict(file_path):
    with open(file_path, 'r') as yaml_file:
        data = yaml.safe_load(yaml_file)
    return data

def save_model_as_pickle(obj, file_path):
    joblib.dump(obj, file_path)

def load_model_from_pickle(file_path):
    return joblib.load(file_path)