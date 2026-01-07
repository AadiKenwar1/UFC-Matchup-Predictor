import xgboost as xgb
import joblib
from tuning import MODEL_PARAMS

# XGBoost model wrapper for UFC fight prediction
class UFCXGBoostModel:
    def __init__(self, **params):
        tuned_params = MODEL_PARAMS
        tuned_params.update(params)
        self.model = xgb.XGBClassifier(**tuned_params)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        else:
            self.model.fit(X_train, y_train, verbose=False)
  
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
    
    def get_feature_importances(self):
        return self.model.feature_importances_
    
    def save(self, filepath):
        joblib.dump(self.model, filepath)
    
    def load(self, filepath):
        self.model = joblib.load(filepath)
