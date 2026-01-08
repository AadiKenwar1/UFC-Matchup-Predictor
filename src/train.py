import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from split_data import temporal_train_test_split
from model import UFCXGBoostModel
from listOfFeatures import FEATURES
from collections import Counter
from tuning import SCALE_POS_WEIGHT


print(len(FEATURES))
# Split the data
print("Splitting data...")
X_train, X_val, X_test, y_train, y_val, y_test = temporal_train_test_split()

print(f"Training set: {len(X_train)} samples")
print(f"Validation set: {len(X_val)} samples")
print(f"Test set: {len(X_test)} samples")

# Calculate class balance
class_counts = Counter(y_train)
scale_pos_weight = SCALE_POS_WEIGHT
print(f"\nClass distribution - Class 0: {class_counts[0]}, Class 1: {class_counts[1]}")
print(f"Using scale_pos_weight: {scale_pos_weight:.3f}")

# Train the model with class weights
print("\nTraining XGBoost model...")
model = UFCXGBoostModel(scale_pos_weight=scale_pos_weight)
model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

# Evaluate on training set
y_train_pred = model.predict(X_train)
y_train_pred_proba = model.predict_proba(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
train_roc_auc = roc_auc_score(y_train, y_train_pred_proba)
print(f"\nTraining - Accuracy: {train_accuracy:.4f}, ROC-AUC: {train_roc_auc:.4f}")

# Evaluate on validation set
y_val_pred = model.predict(X_val)
y_val_pred_proba = model.predict_proba(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
val_roc_auc = roc_auc_score(y_val, y_val_pred_proba)
print(f"\nValidation - Accuracy: {val_accuracy:.4f}, ROC-AUC: {val_roc_auc:.4f}")

# Evaluate on test set
y_test_pred = model.predict(X_test)
y_test_pred_proba = model.predict_proba(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_roc_auc = roc_auc_score(y_test, y_test_pred_proba)
print(f"Test - Accuracy: {test_accuracy:.4f}, ROC-AUC: {test_roc_auc:.4f}")

#Print confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))
#Print classification report
print(f"Classification report:\n {classification_report(y_test, y_test_pred)}")

# Feature importance
print("\nAnalyzing feature importance...")
feature_importance = model.get_feature_importances()
feature_names = X_train.columns
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

# Identify features with zero importance
zero_importance_features = importance_df[importance_df['importance'] == 0.0]['feature'].tolist()
important_features = importance_df[importance_df['importance'] > 0.0]['feature'].tolist()

print(f"\nTotal features: {len(feature_names)}")
print(f"Features with zero importance: {len(zero_importance_features)}")
print(f"Features with non-zero importance: {len(important_features)}")

# Filter out zero-importance features
print("\nFiltering out zero-importance features and retraining...")
X_train_filtered = X_train[important_features]
X_val_filtered = X_val[important_features]
X_test_filtered = X_test[important_features]

# Retrain with filtered features
model_filtered = UFCXGBoostModel(scale_pos_weight=scale_pos_weight)
model_filtered.fit(X_train_filtered, y_train, X_val=X_val_filtered, y_val=y_val)

# Evaluate filtered model
y_test_pred_filtered = model_filtered.predict(X_test_filtered)
y_test_pred_proba_filtered = model_filtered.predict_proba(X_test_filtered)
test_accuracy_filtered = accuracy_score(y_test, y_test_pred_filtered)
test_roc_auc_filtered = roc_auc_score(y_test, y_test_pred_proba_filtered)

print(f"\nFiltered Model - Test Accuracy: {test_accuracy_filtered:.4f}, ROC-AUC: {test_roc_auc_filtered:.4f}")
print(f"Original Model - Test Accuracy: {test_accuracy:.4f}, ROC-AUC: {test_roc_auc:.4f}")

# Print top features
print("\nTop Features (after filtering):")
importance_df_filtered = pd.DataFrame({
    'feature': important_features,
    'importance': importance_df[importance_df['importance'] > 0.0]['importance'].values
}).sort_values('importance', ascending=False)
print(importance_df_filtered.head(0).to_string(index=False))

# Save the filtered model
model_filtered.save('models/ufc_model.pkl')
print(f"\nFiltered model saved to models/ufc_model.pkl ({len(important_features)} features)")