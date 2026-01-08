from pathlib import Path
import pandas as pd
from features import create_features
from model import UFCXGBoostModel
from tuning import SCALE_POS_WEIGHT

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Load all data
print("Loading and creating features...")
df = create_features()

# Remove rows with NaN target (draws)
df = df[df['target'].notna()].copy()

# Separate features and target
X = df.drop(columns=['target', 'DATE'])
y = df['target']

print(f"Training on all data: {len(X)} samples with {len(X.columns)} features")

# First, train a model to get feature importance
# Use last 10% as temporary validation set for early stopping
split_idx = int(len(X) * 0.9)
X_train_temp = X.iloc[:split_idx]
X_val_temp = X.iloc[split_idx:]
y_train_temp = y.iloc[:split_idx]
y_val_temp = y.iloc[split_idx:]

print("\nTraining initial model to identify important features...")
model_initial = UFCXGBoostModel(scale_pos_weight=SCALE_POS_WEIGHT)
model_initial.fit(X_train_temp, y_train_temp, X_val=X_val_temp, y_val=y_val_temp)

# Analyze feature importance
print("\nAnalyzing feature importance...")
feature_importance = model_initial.get_feature_importances()
feature_names = X.columns
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

# Identify features with zero importance
zero_importance_features = importance_df[importance_df['importance'] == 0.0]['feature'].tolist()
important_features = importance_df[importance_df['importance'] > 0.0]['feature'].tolist()

print(f"Total features: {len(feature_names)}")
print(f"Features with zero importance: {len(zero_importance_features)}")
print(f"Features with non-zero importance: {len(important_features)}")

# Filter features
print("\nFiltering out zero-importance features...")
X_filtered = X[important_features]

# Retrain on ALL data with filtered features (no validation set)
print("Retraining final model on all data with filtered features...")
model_final = UFCXGBoostModel(scale_pos_weight=SCALE_POS_WEIGHT)
model_final.fit(X_filtered, y)  # Training on all data

# Save the filtered final model
model_path = str(MODELS_DIR / 'ufc_model_final.pkl')
model_final.save(model_path)
print(f"\nFinal model saved to {model_path} ({len(important_features)} features)")

