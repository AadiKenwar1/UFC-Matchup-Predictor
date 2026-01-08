import pandas as pd
from preprocessor import preprocess_data
from .basic import create_basic_features
from .historical import create_historical_features
from .title_fights import create_title_fight_features
from .ratios import create_ratio_features
from .momentum import create_momentum_features
from .interactions import create_interaction_features
from .consistency import create_consistency_features
from .encoding import create_encoding_features

# Create all features
def create_features():
    df = preprocess_data()
    df = df.copy()
    # Create features in order
    df = create_basic_features(df)
    df = create_historical_features(df)
    df = create_title_fight_features(df)  # Must run after historical (needs fighter1_won_shifted)
    df = create_ratio_features(df)
    df = create_momentum_features(df)
    df = create_interaction_features(df)
    df = create_consistency_features(df)
    df = create_encoding_features(df)
    
    return df

