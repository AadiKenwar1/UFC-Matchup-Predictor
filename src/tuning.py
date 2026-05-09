
VALIDATION_SET_DATE = '2020-01-01'
TEST_SET_DATE = '2024-01-01'

MODEL_PARAMS = {
    'objective': 'binary:logistic',
    'random_state': 42,
    'n_estimators': 400,
    'max_depth': 4,
    'learning_rate': 0.01,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'min_child_weight': 5,
    'reg_lambda': 1.0,
    'gamma': 0.1,
    'reg_alpha': 0.05,
    'early_stopping_rounds': 30,
}

# Computed from training class distribution: class_0 / class_1 = 1705 / 3674 ≈ 0.464
SCALE_POS_WEIGHT = 0.5
