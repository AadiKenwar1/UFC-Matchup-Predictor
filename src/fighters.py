import pandas as pd
from preprocessor import preprocess_data
from features import create_features

# Cache the preprocessed and features data to avoid reloading
_df_preprocessed = None
_df_features = None

#Gets all preprocessed data for the database
def _get_preprocessed_data():
    global _df_preprocessed
    if _df_preprocessed is None:
        _df_preprocessed = preprocess_data()
    return _df_preprocessed

#Gets all features for the database
def _get_features_data():
    global _df_features
    if _df_features is None:
        _df_features = create_features()
    return _df_features

#Gets all fighters in the database
def get_all_fighters():
    df = _get_preprocessed_data()
    fighters = pd.concat([df['fighter1_name'], df['fighter2_name']]).dropna().unique()
    return sorted(fighters.tolist())

#Checks if a fighter exists in the database
def fighter_exists(fighter_name: str) -> bool:
    df = _get_preprocessed_data()
    return fighter_name in df['fighter1_name'].values or fighter_name in df['fighter2_name'].values

#Gets all features for a fighter
def get_fighter_features(fighter_name: str, df_preprocessed: pd.DataFrame = None, df_features: pd.DataFrame = None):
    # Use cached data if not provided
    if df_preprocessed is None:
        df_preprocessed = _get_preprocessed_data()
    if df_features is None:
        df_features = _get_features_data()
    
    # Find fighter's latest fight
    mask = (df_preprocessed['fighter1_name'] == fighter_name) | (df_preprocessed['fighter2_name'] == fighter_name)
    fighter_fights = df_preprocessed[mask]
    
    if len(fighter_fights) == 0:
        raise ValueError(f"Fighter '{fighter_name}' not found")
    
    latest_fight = fighter_fights.sort_values('DATE').iloc[-1]
    feature_row = df_features[df_features['DATE'] == latest_fight['DATE']].iloc[0]
    
    # Determine which fighter position and extract features
    prefix = 'fighter1_' if latest_fight['fighter1_name'] == fighter_name else 'fighter2_'
    fighter_features = feature_row.filter(regex=f'^{prefix}').copy()
    
    # Rename to fighter1_* format if needed
    if prefix == 'fighter2_':
        fighter_features.index = fighter_features.index.str.replace('fighter2_', 'fighter1_')
    
    return fighter_features

