import pandas as pd
import numpy as np
from listOfFeatures import COLS_TO_DROP

def create_encoding_features(df):
    """
    Create target variable and encode categorical features.
    """
    # Create target variable (fighter1_won: 1 if W/L, 0 if L/W, NaN for draws)
    df['target'] = df['OUTCOME'].apply(lambda x: 1 if x == 'W/L' else (0 if x == 'L/W' else np.nan))
    
    # Drop columns we don't want in the final model
    df = df.drop(columns=[col for col in COLS_TO_DROP if col in df.columns])
    
    # Encode categorical variables (one-hot encoding)
    categorical_cols = ['REFEREE', 'WEIGHTCLASS', 'stance_matchup']
    
    # One-hot encode categorical columns, drop first category to avoid multicollinearity
    df = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols, drop_first=True)
    
    return df

