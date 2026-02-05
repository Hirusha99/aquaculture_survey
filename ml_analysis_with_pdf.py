import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("MACHINE LEARNING ANALYSIS: Fish Weight Prediction with PDF Report")
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
for i, (var, cum_var) in enumerate(zip(explained_var, cumulative_var), 1):
    print(f"  PC{i:2d}: {var:6.2%} | Cumulative: {cum_var:6.2%}")

# Determine optimal components
n_components_95 = np.argmax(cumulative_var >= 0.95) + 1

print(f"\nOptimal: {n_components_95} components for 95% variance")

# ============================================================================
# 6. MODEL 2: LINEAR REGRESSION WITH PCA
# ============================================================================
print("\n" + "="*80)
print(f"6. LINEAR REGRESSION WITH PCA ({n_components_95} components)")
print("="*80)

# Apply PCA
pca_optimal = PCA(n_components=n_components_95)
X_train_pca = pca_optimal.fit_transform(X_train_scaled)
X_test_pca = pca_optimal.transform(X_test_scaled)

print(f"  Reduced: {X_train_scaled.shape[1]} → {X_train_pca.shape[1]} features")

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

cv_scores_pca = cross_val_score(lr_pca, X_train_pca, y_train, cv=5, scoring='r2')

print(f"  Test R²: {test_r2_pca:.4f}")
print(f"  Test RMSE: {test_rmse_pca:.4f} kg")

# ============================================================================
# 7. SAVE CSV RESULTS
# ============================================================================
print("\n" + "="*80)
print("7. SAVING CSV RESULTS")
print("="*80)

# Predictions
predictions_df = pd.DataFrame({
    'Actual_Weight': y_test.values,
    'Predicted_LR': y_test_pred,
    'Predicted_LR_PCA': y_test_pred_pca,
    'Error_LR': (y_test.values - y_test_pred),
    'Error_LR_PCA': (y_test.values - y_test_pred_pca)
})
predictions_df.to_csv('model_predictions.csv', index=False)
print("  ✓ model_predictions.csv")

# Feature importance
feature_importance.to_csv('feature_importance.csv', index=False)
print("  ✓ feature_importance.csv")

# PCA components
pca_components_df = pd.DataFrame(
    pca_optimal.components_.T,
    columns=[f'PC{i+1}' for i in range(n_components_95)],
    index=feature_cols
)
pca_components_df.to_csv('pca_components.csv')
print("  ✓ pca_components.csv")

# ============================================================================
# 8. GENERATE PDF REPORT
# ============================================================================
print("\n" + "="*80)
print("8. GENERATING COMPREHENSIVE PDF REPORT")
print("="*80)

pdf_filename = f'ML_Analysis_Report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'

with PdfPages(pdf_filename) as pdf:
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (11, 8.5)

    # ========================================================================
    # PAGE 1: TITLE PAGE
    # ========================================================================
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.5, 0.7, 'Aquaculture Machine Learning Analysis', 
             ha='center', fontsize=28, fontweight='bold')
    fig.text(0.5, 0.6, 'Fish Weight Prediction Model', 
             ha='center', fontsize=20)
    fig.text(0.5, 0.5, f'Linear Regression with PCA Analysis', 
             ha='center', fontsize=16, style='italic')
    fig.text(0.5, 0.35, f'Generated: {datetime.now().strftime("%B %d, %Y at %I:%M %p")}', 
             ha='center', fontsize=12)
    fig.text(0.5, 0.25, f'Dataset: {len(df_clean)} samples | {len(feature_cols)} features', 
             ha='center', fontsize=12)
    fig.text(0.5, 0.15, 'Target Variable: AVG WEIGHT 1 (Fish Final Weight)', 
             ha='center', fontsize=12)
    plt.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # PAGE 2: EXECUTIVE SUMMARY
    # ========================================================================
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')

    summary_text = f"""
EXECUTIVE SUMMARY
{'='*80}

Dataset Overview
  • Total Samples: {len(df_clean)}
  • Training Set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)
  • Testing Set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)
  • Features: {len(feature_cols)}
  • Target: {target} (Fish Final Weight)

Target Variable Statistics
  • Range: {y.min():.3f} - {y.max():.3f} kg
  • Mean: {y.mean():.3f} kg
  • Standard Deviation: {y.std():.3f} kg
  • Median: {y.median():.3f} kg

Model 1: Linear Regression (All Features)
  • Training R²: {train_r2:.4f} ({train_r2*100:.2f}% variance explained)
  • Testing R²: {test_r2:.4f} ({test_r2*100:.2f}% variance explained)
  • Cross-Validation R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}
  • RMSE: {test_rmse:.4f} kg
  • MAE: {test_mae:.4f} kg

Model 2: Linear Regression with PCA
  • Components: {n_components_95} (from {len(feature_cols)} features)
  • Variance Retained: {pca_optimal.explained_variance_ratio_.sum():.2%}
  • Testing R²: {test_r2_pca:.4f} ({test_r2_pca*100:.2f}% variance explained)
  • RMSE: {test_rmse_pca:.4f} kg
  • MAE: {test_mae_pca:.4f} kg

Best Model: {'Linear Regression' if test_r2 > test_r2_pca else 'LR with PCA'}
  • Achieves {max(test_r2, test_r2_pca)*100:.2f}% accuracy on test set
  • Average prediction error: ±{min(test_mae, test_mae_pca):.4f} kg
  • {'Higher accuracy with all features' if test_r2 > test_r2_pca else 'Good accuracy with fewer features'}

Top 3 Most Important Features
  1. {feature_importance.iloc[0]['Feature']}: {feature_importance.iloc[0]['Coefficient']:.4f}
  2. {feature_importance.iloc[1]['Feature']}: {feature_importance.iloc[1]['Coefficient']:.4f}
  3. {feature_importance.iloc[2]['Feature']}: {feature_importance.iloc[2]['Coefficient']:.4f}

Key Insight
  The most influential factor is {feature_importance.iloc[0]['Feature']} with a
  {'positive' if feature_importance.iloc[0]['Coefficient'] > 0 else 'negative'} correlation,
  suggesting that {'higher' if feature_importance.iloc[0]['Coefficient'] > 0 else 'lower'} 
  values lead to {'increased' if feature_importance.iloc[0]['Coefficient'] > 0 else 'decreased'} fish weight.
"""

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace')
    plt.title('Executive Summary', fontsize=16, fontweight='bold', pad=20)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # PAGE 3: MODEL COMPARISON
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

    # R² Comparison
    models = ['Linear\nRegression', 'LR with\nPCA']
    train_scores = [train_r2, train_r2_pca]
    test_scores = [test_r2, test_r2_pca]

    x = np.arange(len(models))
    width = 0.35

    axes[0, 0].bar(x - width/2, train_scores, width, label='Training', color='#667eea')
    axes[0, 0].bar(x + width/2, test_scores, width, label='Testing', color='#764ba2')
    axes[0, 0].set_ylabel('R² Score', fontweight='bold')
    axes[0, 0].set_title('R² Score Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(models)
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)

    # RMSE Comparison
    train_rmse_vals = [train_rmse, train_rmse_pca]
    test_rmse_vals = [test_rmse, test_rmse_pca]

    axes[0, 1].bar(x - width/2, train_rmse_vals, width, label='Training', color='#667eea')
    axes[0, 1].bar(x + width/2, test_rmse_vals, width, label='Testing', color='#764ba2')
    axes[0, 1].set_ylabel('RMSE (kg)', fontweight='bold')
    axes[0, 1].set_title('RMSE Comparison (Lower is Better)')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(models)
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)

    # MAE Comparison
    train_mae_vals = [train_mae, train_mae_pca]
    test_mae_vals = [test_mae, test_mae_pca]

    axes[1, 0].bar(x - width/2, train_mae_vals, width, label='Training', color='#667eea')
    axes[0, 1].bar(x + width/2, test_mae_vals, width, label='Testing', color='#764ba2')
    axes[1, 0].set_ylabel('MAE (kg)', fontweight='bold')
    axes[1, 0].set_title('MAE Comparison (Lower is Better)')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(models)
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)

    # Cross-Validation Scores
    axes[1, 1].boxplot([cv_scores, cv_scores_pca], labels=models)
    axes[1, 1].set_ylabel('R² Score', fontweight='bold')
    axes[1, 1].set_title('Cross-Validation Scores (5-Fold)')
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # PAGE 4: FEATURE IMPORTANCE
    # ========================================================================
    fig, ax = plt.subplots(figsize=(11, 8.5))

    # Plot feature importance
    colors = ['#667eea' if x > 0 else '#764ba2' for x in feature_importance['Coefficient']]
    bars = ax.barh(range(len(feature_importance)), feature_importance['Coefficient'], color=colors)
    ax.set_yticks(range(len(feature_importance)))
    ax.set_yticklabels(feature_importance['Feature'])
    ax.set_xlabel('Coefficient Value', fontweight='bold', fontsize=12)
    ax.set_title('Feature Importance (Linear Regression Coefficients)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (idx, row) in enumerate(feature_importance.iterrows()):
        value = row['Coefficient']
        ax.text(value, i, f' {value:.4f}', va='center', 
                ha='left' if value > 0 else 'right', fontsize=9)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # PAGE 5: ACTUAL VS PREDICTED (LINEAR REGRESSION)
    # ========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
    fig.suptitle('Actual vs Predicted Weight (Linear Regression)', 
                 fontsize=16, fontweight='bold')

    # Training set
    axes[0].scatter(y_train, y_train_pred, alpha=0.6, color='#667eea', s=100, edgecolors='black')
    axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
                 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Weight (kg)', fontweight='bold')
    axes[0].set_ylabel('Predicted Weight (kg)', fontweight='bold')
    axes[0].set_title(f'Training Set (R² = {train_r2:.4f})')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Testing set
    axes[1].scatter(y_test, y_test_pred, alpha=0.6, color='#764ba2', s=100, edgecolors='black')
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                 'r--', lw=2, label='Perfect Prediction')
    axes[1].set_xlabel('Actual Weight (kg)', fontweight='bold')
    axes[1].set_ylabel('Predicted Weight (kg)', fontweight='bold')
    axes[1].set_title(f'Testing Set (R² = {test_r2:.4f})')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # PAGE 6: RESIDUAL ANALYSIS
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle('Residual Analysis (Linear Regression)', fontsize=16, fontweight='bold')

    residuals_train = y_train - y_train_pred
    residuals_test = y_test - y_test_pred

    # Residuals vs Predicted (Training)
    axes[0, 0].scatter(y_train_pred, residuals_train, alpha=0.6, color='#667eea', s=80)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Predicted Weight (kg)', fontweight='bold')
    axes[0, 0].set_ylabel('Residuals (kg)', fontweight='bold')
    axes[0, 0].set_title('Training Set Residuals')
    axes[0, 0].grid(alpha=0.3)

    # Residuals vs Predicted (Testing)
    axes[0, 1].scatter(y_test_pred, residuals_test, alpha=0.6, color='#764ba2', s=80)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Predicted Weight (kg)', fontweight='bold')
    axes[0, 1].set_ylabel('Residuals (kg)', fontweight='bold')
    axes[0, 1].set_title('Testing Set Residuals')
    axes[0, 1].grid(alpha=0.3)

    # Histogram of residuals (Training)
    axes[1, 0].hist(residuals_train, bins=15, color='#667eea', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Residuals (kg)', fontweight='bold')
    axes[1, 0].set_ylabel('Frequency', fontweight='bold')
    axes[1, 0].set_title('Training Residuals Distribution')
    axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].grid(axis='y', alpha=0.3)

    # Histogram of residuals (Testing)
    axes[1, 1].hist(residuals_test, bins=8, color='#764ba2', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Residuals (kg)', fontweight='bold')
    axes[1, 1].set_ylabel('Frequency', fontweight='bold')
    axes[1, 1].set_title('Testing Residuals Distribution')
    axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # PAGE 7: PCA ANALYSIS
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle('Principal Component Analysis', fontsize=16, fontweight='bold')

    # Scree plot
    axes[0, 0].bar(range(1, len(explained_var)+1), explained_var, color='#667eea', alpha=0.7)
    axes[0, 0].set_xlabel('Principal Component', fontweight='bold')
    axes[0, 0].set_ylabel('Variance Explained Ratio', fontweight='bold')
    axes[0, 0].set_title('Scree Plot')
    axes[0, 0].grid(axis='y', alpha=0.3)

    # Cumulative variance
    axes[0, 1].plot(range(1, len(cumulative_var)+1), cumulative_var, 
                    marker='o', color='#764ba2', linewidth=2, markersize=8)
    axes[0, 1].axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
    axes[0, 1].axvline(x=n_components_95, color='g', linestyle='--', 
                       label=f'{n_components_95} components')
    axes[0, 1].set_xlabel('Number of Components', fontweight='bold')
    axes[0, 1].set_ylabel('Cumulative Variance Explained', fontweight='bold')
    axes[0, 1].set_title('Cumulative Variance Explained')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # PCA model predictions vs actual (Training)
    axes[1, 0].scatter(y_train, y_train_pred_pca, alpha=0.6, color='#667eea', s=100)
    axes[1, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
                    'r--', lw=2)
    axes[1, 0].set_xlabel('Actual Weight (kg)', fontweight='bold')
    axes[1, 0].set_ylabel('Predicted Weight (kg)', fontweight='bold')
    axes[1, 0].set_title(f'PCA Model - Training (R² = {train_r2_pca:.4f})')
    axes[1, 0].grid(alpha=0.3)

    # PCA model predictions vs actual (Testing)
    axes[1, 1].scatter(y_test, y_test_pred_pca, alpha=0.6, color='#764ba2', s=100)
    axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                    'r--', lw=2)
    axes[1, 1].set_xlabel('Actual Weight (kg)', fontweight='bold')
    axes[1, 1].set_ylabel('Predicted Weight (kg)', fontweight='bold')
    axes[1, 1].set_title(f'PCA Model - Testing (R² = {test_r2_pca:.4f})')
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # PAGE 8: PREDICTIONS TABLE
    # ========================================================================
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('tight')
    ax.axis('off')

    # Create predictions table
    pred_table_data = []
    pred_table_data.append(['Sample', 'Actual', 'LR Pred', 'LR Error', 'PCA Pred', 'PCA Error'])

    for i, (actual, lr_pred, lr_pca_pred) in enumerate(zip(y_test.values, y_test_pred, y_test_pred_pca), 1):
        pred_table_data.append([
            f'Test {i}',
            f'{actual:.3f}',
            f'{lr_pred:.3f}',
            f'{actual - lr_pred:+.3f}',
            f'{lr_pca_pred:.3f}',
            f'{actual - lr_pca_pred:+.3f}'
        ])

    table = ax.table(cellText=pred_table_data, cellLoc='center', loc='center',
                     colWidths=[0.12, 0.15, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Style header row
    for i in range(6):
        table[(0, i)].set_facecolor('#667eea')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(pred_table_data)):
        for j in range(6):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    plt.title('Test Set Predictions', fontsize=16, fontweight='bold', pad=20)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # PAGE 9: RECOMMENDATIONS
    # ========================================================================
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')

    best_model_name = 'Linear Regression' if test_r2 > test_r2_pca else 'LR with PCA'
    best_r2 = max(test_r2, test_r2_pca)
    best_mae = min(test_mae, test_mae_pca)

    recommendations = f"""
RECOMMENDATIONS & CONCLUSIONS
{'='*80}

Best Model Selection
  ✓ Recommended Model: {best_model_name}
  ✓ Test R² Score: {best_r2:.4f} ({best_r2*100:.2f}% accuracy)
  ✓ Average Error: ±{best_mae:.4f} kg ({best_mae*1000:.1f} grams)
  ✓ Explains {best_r2*100:.2f}% of fish weight variation

Model Interpretation
  The model indicates that fish final weight is primarily influenced by:

  1. {feature_importance.iloc[0]['Feature']} (Coefficient: {feature_importance.iloc[0]['Coefficient']:.4f})
     {'↑ Higher values → Heavier fish' if feature_importance.iloc[0]['Coefficient'] > 0 else '↓ Higher values → Lighter fish'}

  2. {feature_importance.iloc[1]['Feature']} (Coefficient: {feature_importance.iloc[1]['Coefficient']:.4f})
     {'↑ Higher values → Heavier fish' if feature_importance.iloc[1]['Coefficient'] > 0 else '↓ Higher values → Lighter fish'}

  3. {feature_importance.iloc[2]['Feature']} (Coefficient: {feature_importance.iloc[2]['Coefficient']:.4f})
     {'↑ Higher values → Heavier fish' if feature_importance.iloc[2]['Coefficient'] > 0 else '↓ Higher values → Lighter fish'}

Practical Applications
  • Predict final fish weight based on farming conditions
  • Optimize stocking density and feeding strategies
  • Estimate harvest timing for target weights
  • Identify which factors most impact fish growth

Model Limitations
  • Based on {len(df_clean)} historical samples
  • Predictions most reliable within training data range ({y.min():.3f} - {y.max():.3f} kg)
  • Linear relationship assumed between features and weight
  • External factors (disease, weather) not included

PCA Insights
  • Dimensionality reduced from {len(feature_cols)} to {n_components_95} features
  • Retained {pca_optimal.explained_variance_ratio_.sum():.2%} of information
  • {'Simpler model with acceptable accuracy trade-off' if test_r2_pca >= 0.30 else 'Significant accuracy loss with dimensionality reduction'}
  • {'Consider PCA model for deployment if simplicity is priority' if test_r2_pca >= 0.30 else 'Recommend using full feature set for better accuracy'}

Next Steps
  ✓ Deploy {best_model_name} in production
  ✓ Monitor predictions on new data
  ✓ Retrain periodically with updated samples
  ✓ Consider ensemble methods for improved accuracy
  ✓ Collect more data for better generalization

Data Quality Notes
  • {df.shape[0] - len(df_clean)} samples excluded due to missing values
  • DENSITY column had {(df['DENSITY 1 (total stock/Ha)'] == '#REF!').sum()} #REF! errors (cleaned)
  • Consider data collection improvements for future cycles

Report Generated: {datetime.now().strftime("%B %d, %Y at %I:%M %p")}
Analysis Tool: Python with scikit-learn
Models: Linear Regression, PCA, StandardScaler
"""

    ax.text(0.05, 0.95, recommendations, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace')
    plt.title('Recommendations & Conclusions', fontsize=16, fontweight='bold', pad=20)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

print(f"\n  ✓ {pdf_filename}")
print(f"\n{'='*80}")
print(f"PDF REPORT GENERATED SUCCESSFULLY!")
print(f"{'='*80}")
print(f"\nReport saved as: {pdf_filename}")
print(f"Total pages: 9")
print(f"\nReport includes:")
print(f"  • Title page and executive summary")
print(f"  • Model performance comparison")
print(f"  • Feature importance analysis")
print(f"  • Actual vs predicted visualizations")
print(f"  • Residual analysis")
print(f"  • PCA analysis with variance plots")
print(f"  • Detailed predictions table")
print(f"  • Recommendations and conclusions")

print(f"\n{'='*80}")
print(f"ANALYSIS COMPLETE!")
print(f"{'='*80}")
