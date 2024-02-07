from xgboost.sklearn import XGBClassifier
from catboost import CatBoostClassifier
import pandas as pd

class XGBModel():

    def __init__(self, seed, **kwargs):
        self.seed = seed
        self.model = XGBClassifier(random_state=self.seed, **kwargs)
        
    def fit(self, features, labels):
        self.model.fit(features, labels)
        
    def predict(self, features):
        return self.model.predict(features)
    
    def predict_proba(self, features):
        return self.model.predict_proba(features)



class CatBoostModel():

    def __init__(self, seed, **kwargs):
        self.seed = seed
        self.model = CatBoostClassifier(random_state=self.seed, **kwargs)
        
    def fit(self, features, labels):
        self.model.fit(features, labels)
        
    def predict(self, features):
        return self.model.predict(features)
    
    def predict_proba(self, features):
        return self.model.predict_proba(features)