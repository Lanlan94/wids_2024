import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(data, test_size=0.2):
    X = data.drop(columns=['DiagPeriodL90D','patient_id'])
    y = data['DiagPeriodL90D']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

def fill_missing(data):
    return data 

#this one is just to test how we do without any categorical features
def drop_cat(data):
    return data.select_dtypes(include=['number'])

def encode_cat(data,columns = []):
    return data

def normalize_data(data):
    return data

