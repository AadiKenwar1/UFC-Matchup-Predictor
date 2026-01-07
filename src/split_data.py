import pandas as pd
import numpy as np
from features import create_features
from tuning import VALIDATION_SET_DATE, TEST_SET_DATE

# Split data temporally (train on older fights, val on middle, test on newer fights)
def temporal_train_test_split():
    # Create all features (DATE will be included)
    df = create_features()
    # Remove rows with NaN target (draws)
    df = df[df['target'].notna()].copy()
    # Separate features and target
    X = df.drop(columns=['target'])
    y = df['target']
    dates = pd.to_datetime(df['DATE'])
    # Split by date thresholds
    val_split = pd.to_datetime(VALIDATION_SET_DATE)
    test_split = pd.to_datetime(TEST_SET_DATE)
    # Extract rows by date directly
    train_df = df[dates < val_split]
    val_df = df[(dates >= val_split) & (dates < test_split)]
    test_df = df[dates >= test_split]
    # Split the data
    X_train = train_df.drop(columns=['target', 'DATE'])
    X_val = val_df.drop(columns=['target', 'DATE'])
    X_test = test_df.drop(columns=['target', 'DATE'])
    y_train = train_df['target']
    y_val = val_df['target']
    y_test = test_df['target']
    
    return X_train, X_val, X_test, y_train, y_val, y_test

