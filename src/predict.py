import pandas as pd
import numpy as np
from pathlib import Path
from model import UFCXGBoostModel
from fighters import get_fighter_features, _get_preprocessed_data, _get_features_data

# Get project root directory (go up from src/predict.py)
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Cache the model to avoid reloading from disk on every request
_model_cache = None
_model_path_cache = None


def _get_model(model_path: str = None):
    """Load and cache the model (reload only if path changes)"""
    global _model_cache, _model_path_cache

    if model_path is None:
        model_path = str(MODELS_DIR / 'ufc_model_final.pkl')
    else:
        if not Path(model_path).is_absolute():
            model_path = str(MODELS_DIR / model_path)

    if _model_cache is None or _model_path_cache != model_path:
        model = UFCXGBoostModel()
        model.load(model_path)
        _model_cache = model
        _model_path_cache = model_path

    return _model_cache


def _raw_proba_first_fighter_wins(
    first_name: str,
    second_name: str,
    model,
    df_preprocessed: pd.DataFrame,
    df_features: pd.DataFrame,
) -> float:
    """P(model predicts 'first_name' wins) for one corner ordering (first in fighter1 slot)."""
    f1_features = get_fighter_features(first_name, df_preprocessed, df_features)
    f2_features = get_fighter_features(second_name, df_preprocessed, df_features)

    fight_row_dict = f1_features.to_dict()
    for col in f2_features.index:
        fighter2_col = col.replace('fighter1_', 'fighter2_')
        fight_row_dict[fighter2_col] = f2_features[col]

    latest_f1_fight = df_preprocessed[
        (df_preprocessed['fighter1_name'] == first_name)
        | (df_preprocessed['fighter2_name'] == first_name)
    ].sort_values('DATE').iloc[-1]

    latest_f1_features = df_features[df_features['DATE'] == latest_f1_fight['DATE']].iloc[0]
    non_fighter_cols = [
        col
        for col in latest_f1_features.index
        if not col.startswith('fighter') and col not in ['DATE', 'target']
    ]
    for col in non_fighter_cols:
        fight_row_dict[col] = latest_f1_features[col]

    if hasattr(model.model, 'feature_names_in_') and model.model.feature_names_in_ is not None:
        model_feature_names = list(model.model.feature_names_in_)
    else:
        try:
            model_feature_names = model.model.get_booster().feature_names
        except (AttributeError, ValueError, TypeError):
            model_feature_names = [
                col for col in df_features.columns if col not in ['DATE', 'target']
            ]

    for col in model_feature_names:
        if col not in fight_row_dict:
            fight_row_dict[col] = 0

    fight_row = pd.DataFrame([fight_row_dict])
    fight_row = fight_row[model_feature_names].astype(float)
    return float(model.predict_proba(fight_row)[0])


def predict_fight(fighter1_name: str, fighter2_name: str, model_path: str = None):
    """
    Predict win probabilities. Order-invariant: swapping fighter1/fighter2 does not change
    each fighter's win probability (averages both corner orderings for the same model).
    """
    model = _get_model(model_path)
    df_preprocessed = _get_preprocessed_data()
    df_features = _get_features_data()

    p_as_f1 = _raw_proba_first_fighter_wins(
        fighter1_name, fighter2_name, model, df_preprocessed, df_features
    )
    p_as_f2 = _raw_proba_first_fighter_wins(
        fighter2_name, fighter1_name, model, df_preprocessed, df_features
    )
    # Symmetric P(fighter1 wins): average of (prob when f1 in slot1) and (1 - prob when f1 in slot2)
    prob = 0.5 * (p_as_f1 + (1.0 - p_as_f2))
    prob = float(np.clip(prob, 0.0, 1.0))

    return {
        'fighter1': fighter1_name,
        'fighter2': fighter2_name,
        'fighter1_win_probability': round(prob, 4),
        'fighter2_win_probability': round(1.0 - prob, 4),
        'predicted_winner': fighter1_name if prob > 0.5 else fighter2_name,
    }


if __name__ == "__main__":
    result = predict_fight('Merab Dvalishvili', 'Petr Yan')
    swapped = predict_fight('Petr Yan', 'Merab Dvalishvili')
    print("\n" + "=" * 50)
    print("FIGHT PREDICTION (order-invariant)")
    print("=" * 50)
    print(f"\n{result['fighter1']} vs {result['fighter2']}")
    print(f"\nPredicted Winner: {result['predicted_winner']}")
    print(f"\nWin Probabilities:")
    print(f"  {result['fighter1']}: {result['fighter1_win_probability']:.1%}")
    print(f"  {result['fighter2']}: {result['fighter2_win_probability']:.1%}")
    print("\nSwapped order — Merab win % should match fighter2 above:")
    print(f"  Merab: {swapped['fighter1_win_probability']:.1%}  Petr: {swapped['fighter2_win_probability']:.1%}")
    print("=" * 50)
