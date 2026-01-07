from features import create_features
from model import UFCXGBoostModel
from tuning import SCALE_POS_WEIGHT

# Load all data
print("Loading and creating features...")
df = create_features()

# Remove rows with NaN target (draws)
df = df[df['target'].notna()].copy()

# Separate features and target
X = df.drop(columns=['target', 'DATE'])
y = df['target']

print(f"Training on all data: {len(X)} samples")

# Train model on all data
print("Training XGBoost model on all data...")
model = UFCXGBoostModel(scale_pos_weight=SCALE_POS_WEIGHT)
model.fit(X, y)  # No validation set - training on all data

# Save the model
model.save('models/ufc_model_final.pkl')
print(f"\nModel saved to models/ufc_model_final.pkl ({len(X.columns)} features)")

