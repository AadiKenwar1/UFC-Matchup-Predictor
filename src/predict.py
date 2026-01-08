import pandas as pd
import numpy as np
from model import UFCXGBoostModel
from fighters import get_fighter_features, _get_preprocessed_data, _get_features_data

# Cache the model to avoid reloading from disk on every request
_model_cache = None
_model_path_cache = None

def _get_model(model_path: str = 'models/ufc_model_final.pkl'):
    """Load and cache the model (reload only if path changes)"""
    global _model_cache, _model_path_cache
    
    if _model_cache is None or _model_path_cache != model_path:
        model = UFCXGBoostModel()
        model.load(model_path)
        _model_cache = model
        _model_path_cache = model_path
    
    return _model_cache

def predict_fight(fighter1_name: str, fighter2_name: str, model_path: str = 'models/ufc_model_final.pkl'):
    # Load cached model (only loads from disk once)
    model = _get_model(model_path)
    
    # Get full datasets (with all historical data for proper feature calculation)
    df_preprocessed = _get_preprocessed_data()
    df_features = _get_features_data()
    
    # Get each fighter's latest features (normalized to fighter1_* format)
    f1_features = get_fighter_features(fighter1_name, df_preprocessed, df_features)
    f2_features = get_fighter_features(fighter2_name, df_preprocessed, df_features)
    
    # Build prediction row as dictionary (avoids DataFrame fragmentation)
    fight_row_dict = f1_features.to_dict()
    
    # Add fighter2 features (rename fighter1_* to fighter2_*)
    for col in f2_features.index:
        fighter2_col = col.replace('fighter1_', 'fighter2_')
        fight_row_dict[fighter2_col] = f2_features[col]
    
    # Get non-fighter-specific features from fighter1's latest fight
    latest_f1_fight = df_preprocessed[
        (df_preprocessed['fighter1_name'] == fighter1_name) | 
        (df_preprocessed['fighter2_name'] == fighter1_name)
    ].sort_values('DATE').iloc[-1]
    
    latest_f1_features = df_features[df_features['DATE'] == latest_f1_fight['DATE']].iloc[0]
    
    # Add non-fighter-specific features (month, is_title_fight, etc.)
    non_fighter_cols = [col for col in latest_f1_features.index 
                       if not col.startswith('fighter') and col not in ['DATE', 'target']]
    for col in non_fighter_cols:
        fight_row_dict[col] = latest_f1_features[col]
    
    # Get model's expected feature names
    if hasattr(model.model, 'feature_names_in_') and model.model.feature_names_in_ is not None:
        model_feature_names = list(model.model.feature_names_in_)
    else:
        try:
            model_feature_names = model.model.get_booster().feature_names
        except:
            model_feature_names = [col for col in df_features.columns if col not in ['DATE', 'target']]
    
    # Ensure all features exist (fill missing with 0)
    for col in model_feature_names:
        if col not in fight_row_dict:
            fight_row_dict[col] = 0
    
    # Create DataFrame once from dictionary (no fragmentation)
    fight_row = pd.DataFrame([fight_row_dict])
    
    # Convert to numeric and select features in correct order
    fight_row = fight_row[model_feature_names].astype(float)
    
    # Make prediction
    prob = model.predict_proba(fight_row)[0]
    
    return {
        'fighter1': fighter1_name,
        'fighter2': fighter2_name,
        'fighter1_win_probability': round(float(prob), 4),
        'fighter2_win_probability': round(float(1 - prob), 4),
        'predicted_winner': fighter1_name if prob > 0.5 else fighter2_name
    }


if __name__ == "__main__":
    result = predict_fight('Merab Dvalishvili', 'Petr Yan')
    print("\n" + "="*50)
    print("FIGHT PREDICTION")
    print("="*50)
    print(f"\n{result['fighter1']} vs {result['fighter2']}")
    print(f"\nPredicted Winner: {result['predicted_winner']}")
    print(f"\nWin Probabilities:")
    print(f"  {result['fighter1']}: {result['fighter1_win_probability']:.1%}")
    print(f"  {result['fighter2']}: {result['fighter2_win_probability']:.1%}")
    print("="*50)
