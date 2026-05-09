import xgboost as xgb
import joblib
from tuning import MODEL_PARAMS

# XGBoost model wrapper for UFC fight prediction
class UFCXGBoostModel:
    def __init__(self, **params):
        merged = dict(MODEL_PARAMS)
        merged.update(params)
        # XGBoost 3.x: early_stopping_rounds on the estimator requires eval_set on every fit().
        # trainFinal.py fits on all data without a val set — only use ES when val is passed to fit().
        self._early_stopping_rounds = merged.pop("early_stopping_rounds", None)
        self._clf_params = merged
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if X_val is not None and y_val is not None and self._early_stopping_rounds:
            self.model = xgb.XGBClassifier(
                **{**self._clf_params, "early_stopping_rounds": self._early_stopping_rounds}
            )
            self.model.fit(
                X_train, y_train, eval_set=[(X_val, y_val)], verbose=False
            )
        else:
            self.model = xgb.XGBClassifier(**self._clf_params)
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
