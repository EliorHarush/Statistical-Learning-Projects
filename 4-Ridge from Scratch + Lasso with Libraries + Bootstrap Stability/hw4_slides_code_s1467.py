import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# ==========================================
# 1. LOAD DATA & MODEL RESULTS
# ==========================================
print("Loading data and model results...")
train_df = pd.read_csv('hw4_train_s1467.csv')
train_df = train_df.loc[:, ~train_df.columns.str.contains('^Unnamed')]
X = train_df.drop(columns=['y'])
y = train_df['y']

with open('hw4_results_s1467.json', 'r') as f:
    results = json.load(f)

ridge_betas = np.array(results['ridge_beta'])
lasso_betas = np.array(results['lasso_beta'])
ridge_lambda = results['ridge_lambda']
lasso_alpha = results['lasso_alpha']
n_nonzero = results['lasso_n_nonzero']
ridge_test_pred = np.array(results['ridge_pred_test'])
lasso_test_pred = np.array(results['lasso_pred_test'])
ridge_cv_mse = np.array(results['ridge_cv_mse'])
boot_freqs = np.array(results['bootstrap_selection_freq'])

predictors = np.arange(1, 41)

# ==========================================
# 2. MASTER PDF EXPORT
# ==========================================
pdf_filename = 'hw4_slides_s1467.pdf'

with PdfPages(pdf_filename) as pdf:
    
    # ------------------------------------------
    # SLIDE 1: DATA
    # ------------------------------------------
    fig1 = plt.figure(figsize=(16, 9), facecolor='white')
    fig1.suptitle('Data', fontsize=26, fontweight='bold', y=0.95)
    
    gs1 = gridspec.GridSpec(2, 3, width_ratios=[1, 1.2, 1.2], hspace=0.4, wspace=0.3)
    
    # LEFT: Summary Stats
    ax_stats = fig1.add_subplot(gs1[:, 0])
    ax_stats.axis('off')
    stats_text = (
        "Summary Statistics\n"
        "-------------------------\n"
        f"Observations (n) : {len(train_df)}\n"
        f"Predictors (p)   : {X.shape[1]}\n\n"
        f"Target (y) Mean  : {y.mean():.2f}\n"
        f"Target (y) Std   : {y.std():.2f}\n\n"
        "Key Observation:\n"
        "High-dimensional space with\n"
        "heavy block-correlation requires\n"
        "regularized model fitting."
    )
    ax_stats.text(0.05, 0.5, stats_text, fontsize=16, va='center', ha='left')

    # MIDDLE TOP: Top 10 Predictors
    ax_bar1 = fig1.add_subplot(gs1[0, 1])
    corrs_with_y = train_df.corr()['y'].drop('y').abs().sort_values(ascending=False).head(10)
    sns.barplot(x=corrs_with_y.values, y=corrs_with_y.index, ax=ax_bar1, palette='viridis')
    ax_bar1.set_title('Top 10 Predictors (Correlation w/ y)', fontweight='bold')
    ax_bar1.set_xlabel('Absolute Correlation')

    # MIDDLE BOTTOM: Target Distribution
    ax_dist = fig1.add_subplot(gs1[1, 1])
    sns.histplot(y, kde=True, ax=ax_dist, color='navy')
    ax_dist.set_title('Distribution of Target (y)', fontweight='bold')
    ax_dist.set_xlabel('y values')
    ax_dist.set_ylabel('Frequency')


    # RIGHT: Correlation Heatmap
    ax_heat = fig1.add_subplot(gs1[:, 2])
    corr_matrix = X.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0, 
                xticklabels=False, yticklabels=False, cbar_kws={"shrink": .8}, square=True, ax=ax_heat)
    ax_heat.set_title('Predictor Correlation Heatmap', fontweight='bold')
    
    plt.subplots_adjust(top=0.85, bottom=0.1, left=0.05, right=0.95)
    pdf.savefig(fig1)
    plt.close(fig1)

    # ------------------------------------------
    # SLIDE 2: METHODOLOGY & RESULTS
    # ------------------------------------------
    fig2 = plt.figure(figsize=(16, 9), facecolor='white')
    fig2.suptitle('Methodology & Results', fontsize=26, fontweight='bold', y=0.95)
    
    gs2 = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.1)
    
    # LEFT: Methodology & Results
    ax_meth = fig2.add_subplot(gs2[0])
    ax_meth.axis('off')
    ridge_mean_pred = np.mean(ridge_test_pred)
    lasso_mean_pred = np.mean(lasso_test_pred)
    
    ridge_mean_pred = np.mean(ridge_test_pred)
    lasso_mean_pred = np.mean(lasso_test_pred)

    ridge_var_pred = np.var(ridge_test_pred)
    lasso_var_pred = np.var(lasso_test_pred)

    meth_text = (
        "Methodology\n"
        "-------------------------\n"
        "• Standardized all predictors to ensure\n"
        "  fair penalization.\n"
        "• Fit models using 5-Fold Cross-Validation\n"
        "  to isolate optimal penalties.\n"
        "• Evaluated both L2 (Ridge) and L1 (Lasso)\n"
        "  regularization techniques.\n\n"
        "Core Results\n"
        "-------------------------\n"
        f"• Ridge Optimal Lambda: {ridge_lambda:.4f}\n"
        "• Ridge variables retained: 40 (100%)\n"
        f"• Ridge mean test prediction: {ridge_mean_pred:.2f}\n"
        f"• Ridge test prediction variance: {ridge_var_pred:.2f}\n\n"
        f"• Lasso Optimal Alpha: {lasso_alpha:.4f}\n"
        f"• Lasso variables retained: {n_nonzero} ({n_nonzero/40*100:.1f}%)\n"
        f"• Lasso mean test prediction: {lasso_mean_pred:.2f}\n"
        f"• Lasso test prediction variance: {lasso_var_pred:.2f}\n"
    )
    ax_meth.text(0.05, 0.5, meth_text, fontsize=15, va='center', ha='left')

    # MIDDLE: Explicit Ridge Model
    ax_ridge_mod = fig2.add_subplot(gs2[1])
    ax_ridge_mod.axis('off')
    top_r = np.argsort(np.abs(ridge_betas))[::-1][:15] 
    r_eq = "\n".join([f"  + ({ridge_betas[i]:.4f} * x{i+1})" for i in top_r])
    
    ridge_model_text = (
        "Final Ridge Model (Top 15)\n"
        "-------------------------\n"
        "y_pred = intercept\n"
        f"{r_eq}\n"
        "  ... (+ 25 additional variables)"
    )
    ax_ridge_mod.text(0.1, 0.5, ridge_model_text, fontsize=14, va='center', ha='left')

    # RIGHT: Explicit Lasso Model
    ax_lasso_mod = fig2.add_subplot(gs2[2])
    ax_lasso_mod.axis('off')
    nonzero_idx = np.where(np.abs(lasso_betas) > 1e-8)[0]
    l_eq = "\n".join([f"  + ({lasso_betas[i]:.4f} * x{i+1})" for i in nonzero_idx])
    
    lasso_model_text = (
        "Final Lasso Model (Complete)\n"
        "-------------------------\n"
        "y_pred = intercept\n"
        f"{l_eq}"
    )
    ax_lasso_mod.text(0.1, 0.5, lasso_model_text, fontsize=14, va='center', ha='left')

    plt.subplots_adjust(top=0.85, bottom=0.1, left=0.05, right=0.95)
    pdf.savefig(fig2)
    plt.close(fig2)

    # ------------------------------------------
    # SLIDE 3: VISUALIZATION (100% Visuals, 4 Graphs)
    # ------------------------------------------
    fig3 = plt.figure(figsize=(16, 9), facecolor='white')
    fig3.suptitle('Visualization', fontsize=26, fontweight='bold', y=0.95)
    
    # 2x2 Grid for 4 distinct charts
    gs3 = gridspec.GridSpec(2, 2, wspace=0.25, hspace=0.35)
    
    # --- VISUAL 1: METHODOLOGY (Ridge CV Curve) ---
    ax_cv = fig3.add_subplot(gs3[0, 0])
    ax_cv.plot(ridge_cv_mse, color='navy', lw=2, marker='.', markersize=8)
    ax_cv.set_title('Methodology: Ridge Optimization (CV MSE Curve)', fontweight='bold', fontsize=14)
    ax_cv.set_xlabel('Lambda Grid Search Index', fontsize=12)
    ax_cv.set_ylabel('Mean Squared Error', fontsize=12)
    ax_cv.grid(True, linestyle='--', alpha=0.6)

    # --- VISUAL 2: RESULTS (Coefficient Shrinkage vs Sparsity) ---
    ax_coef = fig3.add_subplot(gs3[0, 1])
    ax_coef.scatter(ridge_betas, lasso_betas, color='purple', alpha=0.7, edgecolor='black', s=60)
    ax_coef.axhline(0, color='black', linewidth=1)
    ax_coef.axvline(0, color='black', linewidth=1)
    ax_coef.set_title('Results: Ridge vs. Lasso Coefficient Comparison', fontweight='bold', fontsize=14)
    ax_coef.set_xlabel('Ridge Coefficients (Shrunk)', fontsize=12)
    ax_coef.set_ylabel('Lasso Coefficients (Sparse)', fontsize=12)
    ax_coef.grid(True, linestyle='--', alpha=0.6)

    # --- VISUAL 3: PREDICTIONS (Model Divergence) ---
    ax_pred = fig3.add_subplot(gs3[1, 0])
    ax_pred.scatter(ridge_test_pred, lasso_test_pred, color='teal', alpha=0.7, edgecolor='black', s=40)
    
    # Add a y=x reference line
    min_val = min(ridge_test_pred.min(), lasso_test_pred.min())
    max_val = max(ridge_test_pred.max(), lasso_test_pred.max())
    ax_pred.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Agreement (y=x)')
    
    ax_pred.set_title('Results: Test Set Prediction Variance', fontweight='bold', fontsize=14)
    ax_pred.set_xlabel('Ridge Predicted Responses (y)', fontsize=12)
    ax_pred.set_ylabel('Lasso Predicted Responses (y)', fontsize=12)
    ax_pred.legend(loc='upper left')
    ax_pred.grid(True, linestyle='--', alpha=0.6)

    # --- VISUAL 4: THE FLAW (Bootstrap Stability) ---
    ax_boot = fig3.add_subplot(gs3[1, 1])
    
    # Color variables: Red if highly unstable (< 0.8), Navy if stable (> 0.8)
    boot_colors = ['navy' if f > 0.8 else 'lightcoral' for f in boot_freqs]
    ax_boot.bar(predictors, boot_freqs, color=boot_colors, edgecolor='black')
    
    ax_boot.axhline(0.8, color='black', linestyle='--', linewidth=1.5, label='80% Stability Threshold')
    ax_boot.set_title('Lasso Flaw: Bootstrap Selection Instability', fontweight='bold', fontsize=14)
    ax_boot.set_xlabel('Predictors (x1 to x40)', fontsize=12)
    ax_boot.set_ylabel('Selection Frequency (0.0 to 1.0)', fontsize=12)
    ax_boot.set_ylim(0, 1.05)
    ax_boot.legend(loc='upper right')

    plt.subplots_adjust(top=0.88, bottom=0.08, left=0.05, right=0.95)
    pdf.savefig(fig3)
    plt.close(fig3)

