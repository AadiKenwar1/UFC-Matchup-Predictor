
VALIDATION_SET_DATE = '2020-01-01'
TEST_SET_DATE = '2024-01-01'

MODEL_PARAMS = {
    'objective': 'binary:logistic',
    'random_state': 42,
    'n_estimators': 140,
    'max_depth': 4,
    'learning_rate': 0.01,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'min_child_weight': 5,         
    'reg_lambda': 1.15,
    'gamma': 0.1,
    'reg_alpha': 0.05
}

SCALE_POS_WEIGHT = 0.6