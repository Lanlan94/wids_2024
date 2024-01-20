import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(data, test_size=0.2):
    X = data.drop(columns=['DiagPeriodL90D'])
    y = data['DiagPeriodL90D']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    return X_train, X_test, y_train, y_test

def get_fill_for_na(data, columns=[], fill_strategy='separate_category'):
    
    filler_values = {}

    for column in columns:
        
        if fill_strategy == 'mean':
            fill_value = data[column].mean()   

        elif fill_strategy == 'median':
            fill_value = data[column].median()
                
        elif fill_strategy == 'mode':
            fill_value = data[column].mode()[0]
        
        elif fill_strategy == 'separate_category':
            fill_value = 'nan'

        filler_values[column] = fill_value

    return filler_values


def drop_cat(data):
    #this one is just to test how we do without any categorical features
    return data.select_dtypes(include=['number'])

def encode_cat(data,columns = []):
    return data

def normalize_data(data):
    return data

