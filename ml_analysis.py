import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("MACHINE LEARNING ANALYSIS: Fish Weight Prediction")
print("="*80)

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================
print("\n1. Loading Data...")

df = pd.read_csv('data/survey data.csv')

# Clean DENSITY column
df['DENSITY 1 (total stock/Ha)'] = df['DENSITY 1 (total stock/Ha)'].replace('#REF!', np.nan)
df['DENSITY 1 (total stock/Ha)'] = pd.to_numeric(df['DENSITY 1 (total stock/Ha)'])

# Define target and features
target = 'AVG WEIGHT 1'
feature_cols = [
    'Area (Ha 1)', 
    'INITIAL WEIGHT/kg', 
    'DENSITY 1 (total stock/Ha)',
    'HARVEST NO. 1', 
    'PRODUCTION 1', 
    'PRODUCTION 1/Ha',
    'DISTANCE/m', 
    'VEGETATION%', 
    'AVERAGE_DEPTH/ft',
    'PENS', 
    'DURATIONMONTHS'
]

# Create clean dataset
df_clean = df[feature_cols + [target]].dropna()
X = df_clean[feature_cols]
y = df_clean[target]

print(f"  Dataset: {df_clean.shape[0]} samples, {len(feature_cols)} features")
print(f"  Target: {target}")
print(f"    Range: {y.min():.3f} - {y.max():.3f} kg")
print(f"    Mean: {y.mean():.3f} ± {y.std():.3f} kg")

# ============================================================================
# 2. SPLIT DATA
# ============================================================================
print("\n2. Splitting Data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"  Training: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Testing: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

# ============================================================================
# 3. FEATURE SCALING
# ============================================================================
print("\n3. Scaling Features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("  ✓ Features standardized (mean=0, std=1)")

# ============================================================================
# 4. MODEL 1: LINEAR REGRESSION (All Features)
# ============================================================================
print("\n" + "="*80)
print("4. LINEAR REGRESSION MODEL (All Features)")
print("="*80)

# Train model
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Predictions
y_train_pred = lr_model.predict(X_train_scaled)
y_test_pred = lr_model.predict(X_test_scaled)

# Metrics
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

# Cross-validation
cv_scores = cross_val_score(lr_model, X_train_scaled, y_train, cv=5, scoring='r2')

print("\nPerformance Metrics:")
print(f"  R² Score:")
print(f"    Training: {train_r2:.4f}")
print(f"    Testing:  {test_r2:.4f}")
print(f"    CV Mean:  {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
print(f"\n  RMSE (Root Mean Squared Error):")
print(f"    Training: {train_rmse:.4f} kg")
print(f"    Testing:  {test_rmse:.4f} kg")
print(f"\n  MAE (Mean Absolute Error):")
print(f"    Training: {train_mae:.4f} kg")
print(f"    Testing:  {test_mae:.4f} kg")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': lr_model.coef_,
    'Abs_Coefficient': np.abs(lr_model.coef_)
}).sort_values('Abs_Coefficient', ascending=False)

print("\nFeature Importance (Ranked by Absolute Coefficient):")
for idx, row in feature_importance.iterrows():
    direction = "↑" if row['Coefficient'] > 0 else "↓"
    print(f"  {row['Feature']:30s}: {row['Coefficient']:8.4f} {direction}")

# Model equation
print(f"\nModel Intercept: {lr_model.intercept_:.4f}")

# ============================================================================
# 5. PRINCIPAL COMPONENT ANALYSIS (PCA)
# ============================================================================
print("\n" + "="*80)
print("5. PRINCIPAL COMPONENT ANALYSIS (PCA)")
print("="*80)

# Perform PCA
pca_full = PCA()
pca_full.fit(X_train_scaled)

# Variance explained
explained_var = pca_full.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

print("\nVariance Explained by Each Component:")
print("-" * 60)
for i, (var, cum_var) in enumerate(zip(explained_var, cumulative_var), 1):
    bar = "█" * int(var * 50)
    print(f"  PC{i:2d}: {var:6.2%} {bar:20s} | Cumulative: {cum_var:6.2%}")

# Determine optimal components
n_components_80 = np.argmax(cumulative_var >= 0.80) + 1
n_components_90 = np.argmax(cumulative_var >= 0.90) + 1
n_components_95 = np.argmax(cumulative_var >= 0.95) + 1

print(f"\nOptimal Number of Components:")
print(f"  For 80% variance: {n_components_80} components")
print(f"  For 90% variance: {n_components_90} components")
print(f"  For 95% variance: {n_components_95} components")

# ============================================================================
# 6. MODEL 2: LINEAR REGRESSION WITH PCA
# ============================================================================
print("\n" + "="*80)
print(f"6. LINEAR REGRESSION WITH PCA ({n_components_95} components for 95% variance)")
print("="*80)

# Apply PCA
pca_optimal = PCA(n_components=n_components_95)
X_train_pca = pca_optimal.fit_transform(X_train_scaled)
X_test_pca = pca_optimal.transform(X_test_scaled)

print(f"\nDimensionality Reduction:")
print(f"  Original: {X_train_scaled.shape[1]} features")
print(f"  Reduced:  {X_train_pca.shape[1]} components")
print(f"  Variance: {pca_optimal.explained_variance_ratio_.sum():.2%}")
print(f"  Reduction: {(1 - X_train_pca.shape[1]/X_train_scaled.shape[1])*100:.1f}%")

# Train model
lr_pca = LinearRegression()
lr_pca.fit(X_train_pca, y_train)

# Predictions
y_train_pred_pca = lr_pca.predict(X_train_pca)
y_test_pred_pca = lr_pca.predict(X_test_pca)

# Metrics
train_r2_pca = r2_score(y_train, y_train_pred_pca)
test_r2_pca = r2_score(y_test, y_test_pred_pca)
train_rmse_pca = np.sqrt(mean_squared_error(y_train, y_train_pred_pca))
test_rmse_pca = np.sqrt(mean_squared_error(y_test, y_test_pred_pca))
train_mae_pca = mean_absolute_error(y_train, y_train_pred_pca)
test_mae_pca = mean_absolute_error(y_test, y_test_pred_pca)

# Cross-validation
cv_scores_pca = cross_val_score(lr_pca, X_train_pca, y_train, cv=5, scoring='r2')

print("\nPerformance Metrics:")
print(f"  R² Score:")
print(f"    Training: {train_r2_pca:.4f}")
print(f"    Testing:  {test_r2_pca:.4f}")
print(f"    CV Mean:  {cv_scores_pca.mean():.4f} (±{cv_scores_pca.std():.4f})")
print(f"\n  RMSE:")
print(f"    Training: {train_rmse_pca:.4f} kg")
print(f"    Testing:  {test_rmse_pca:.4f} kg")
print(f"\n  MAE:")
print(f"    Training: {train_mae_pca:.4f} kg")
print(f"    Testing:  {test_mae_pca:.4f} kg")

# PCA component loadings
print("\nTop Contributing Features for Each Component:")
pca_components_df = pd.DataFrame(
    pca_optimal.components_.T,
    columns=[f'PC{i+1}' for i in range(n_components_95)],
    index=feature_cols
)

for i in range(min(3, n_components_95)):  # Show first 3 components
    print(f"\n  Principal Component {i+1} (explains {explained_var[i]:.1%}):")
    component_contributions = pca_components_df.iloc[:, i].abs().sort_values(ascending=False)
    for feature, value in component_contributions.head(3).items():
        print(f"    • {feature}: {value:.3f}")

# ============================================================================
# 7. MODEL COMPARISON
# ============================================================================
print("\n" + "="*80)
print("7. MODEL COMPARISON")
print("="*80)

comparison = pd.DataFrame({
    'Metric': ['R² (Train)', 'R² (Test)', 'R² (CV)', 'RMSE (Train)', 'RMSE (Test)', 
               'MAE (Train)', 'MAE (Test)', 'Features'],
    'Linear Regression': [
        f"{train_r2:.4f}", f"{test_r2:.4f}", f"{cv_scores.mean():.4f}",
        f"{train_rmse:.4f}", f"{test_rmse:.4f}",
        f"{train_mae:.4f}", f"{test_mae:.4f}",
        str(X_train_scaled.shape[1])
    ],
    'LR with PCA': [
        f"{train_r2_pca:.4f}", f"{test_r2_pca:.4f}", f"{cv_scores_pca.mean():.4f}",
        f"{train_rmse_pca:.4f}", f"{test_rmse_pca:.4f}",
        f"{train_mae_pca:.4f}", f"{test_mae_pca:.4f}",
        str(X_train_pca.shape[1])
    ]
})

print("\n", comparison.to_string(index=False))

# Determine best model
best_model = "Linear Regression" if test_r2 > test_r2_pca else "LR with PCA"
print(f"\n✓ Best Model: {best_model} (Higher Test R²)")

# ============================================================================
# 8. SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("8. SAVING RESULTS")
print("="*80)

# Predictions
predictions_df = pd.DataFrame({
    'Actual_Weight': y_test.values,
    'Predicted_LR': y_test_pred,
    'Predicted_LR_PCA': y_test_pred_pca,
    'Error_LR': (y_test.values - y_test_pred),
    'Error_LR_PCA': (y_test.values - y_test_pred_pca),
    'Abs_Error_LR': np.abs(y_test.values - y_test_pred),
    'Abs_Error_LR_PCA': np.abs(y_test.values - y_test_pred_pca)
})
predictions_df.to_csv('model_predictions.csv', index=False)
print("  ✓ model_predictions.csv")

# Feature importance
feature_importance.to_csv('feature_importance.csv', index=False)
print("  ✓ feature_importance.csv")

# PCA components
pca_components_df.to_csv('pca_components.csv')
print("  ✓ pca_components.csv")

# Model metrics summary
metrics_df = pd.DataFrame({
    'Model': ['Linear Regression', 'LR with PCA'],
    'Train_R2': [train_r2, train_r2_pca],
    'Test_R2': [test_r2, test_r2_pca],
    'CV_R2_Mean': [cv_scores.mean(), cv_scores_pca.mean()],
    'CV_R2_Std': [cv_scores.std(), cv_scores_pca.std()],
    'Train_RMSE': [train_rmse, train_rmse_pca],
    'Test_RMSE': [test_rmse, test_rmse_pca],
    'Train_MAE': [train_mae, train_mae_pca],
    'Test_MAE': [test_mae, test_mae_pca],
    'Features': [X_train_scaled.shape[1], X_train_pca.shape[1]]
})
metrics_df.to_csv('model_metrics.csv', index=False)
print("  ✓ model_metrics.csv")

# PCA variance explained
variance_df = pd.DataFrame({
    'Component': [f'PC{i+1}' for i in range(len(explained_var))],
    'Variance_Explained': explained_var,
    'Cumulative_Variance': cumulative_var
})
variance_df.to_csv('pca_variance.csv', index=False)
print("  ✓ pca_variance.csv")

# ============================================================================
# 9. SUMMARY
# ============================================================================
print("\n" + "="*80)
print("9. SUMMARY & INSIGHTS")
print("="*80)

print(f"\nTarget Variable: {target}")
print(f"  • Predicting fish final weight (kg)")
print(f"  • Range: {y.min():.3f} - {y.max():.3f} kg")

print(f"\nBest Model: {best_model}")
print(f"  • Test R²: {max(test_r2, test_r2_pca):.4f}")
print(f"  • Explains {max(test_r2, test_r2_pca)*100:.2f}% of weight variation")
print(f"  • Average error: {min(test_mae, test_mae_pca):.4f} kg")

print(f"\nTop 3 Most Important Features:")
for i, row in feature_importance.head(3).iterrows():
    print(f"  {i+1}. {row['Feature']}: {row['Coefficient']:.4f}")

print(f"\nPCA Results:")
print(f"  • Reduced {X.shape[1]} features → {n_components_95} components")
print(f"  • Retained {pca_optimal.explained_variance_ratio_.sum():.1%} of variance")
print(f"  • Simpler model, but {'lower' if test_r2_pca < test_r2 else 'higher'} accuracy")

print(f"\nRecommendation:")
if test_r2 > test_r2_pca:
    print("  → Use Linear Regression (all features) for best accuracy")
else:
    print("  → Use PCA model for simpler deployment with good accuracy")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
