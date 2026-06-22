import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.utils.class_weight import compute_sample_weight

# ==========================================
# 1. UTILITY CODE: DATA, RESULTS & METRICS
# ==========================================
student_id = "s1467"

print("Loading data and JSON results...")
train_df = pd.read_csv(f'hw6_train_{student_id}.csv')
test_df = pd.read_csv(f'hw6_test_{student_id}.csv')

with open(f'hw6_results_{student_id}.json', 'r') as f:
    res = json.load(f)

# Extract Training Data
X_train = train_df.drop(columns=['sale_price_keur', 'distressed']).values
y_reg = train_df['sale_price_keur'].values
y_clf = train_df['distressed'].values
feature_names = train_df.drop(columns=['sale_price_keur', 'distressed']).columns

print("Calculating performance metrics for Slide 2 and 3 comparisons...")
# --- Regression Models (Metric: Mean Squared Error) ---
best_depth = res['tree_max_depth']

# 1. CART
tree_reg = DecisionTreeRegressor(max_depth=best_depth, random_state=42)
tree_reg.fit(X_train, y_reg)
mse_cart = mean_squared_error(y_reg, tree_reg.predict(X_train))

# 2. Random Forest
rf_reg = RandomForestRegressor(n_estimators=res['rf_reg_n_trees'], max_features=res['rf_reg_mtry'], max_depth=best_depth, random_state=42)
rf_reg.fit(X_train, y_reg)
mse_rf = mean_squared_error(y_reg, rf_reg.predict(X_train))

# 3. GBM
gbm_reg = GradientBoostingRegressor(**res['gbm_reg_hyperparams'], random_state=42)
gbm_reg.fit(X_train, y_reg)
mse_gbm = mean_squared_error(y_reg, gbm_reg.predict(X_train))

# --- Classification Models (Metric: Cross-Entropy / Log Loss) ---
# 4. Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=res['rf_clf_n_trees'], max_features=res['rf_clf_mtry'], max_depth=res['rf_clf_max_depth'], class_weight='balanced', random_state=42)
rf_clf.fit(X_train, y_clf)
loss_rf = log_loss(y_clf, rf_clf.predict_proba(X_train))

# 5. GBM Classifier
sw = compute_sample_weight('balanced', y_clf)
gbm_clf = GradientBoostingClassifier(**res['gbm_clf_hyperparams'], random_state=42)
gbm_clf.fit(X_train, y_clf, sample_weight=sw)
loss_gbm = log_loss(y_clf, gbm_clf.predict_proba(X_train))

# ==========================================
# 2. SLIDE GENERATION
# ==========================================
pdf_filename = f"hw6_slides_{student_id}.pdf"
print(f"Generating {pdf_filename}...")

sns.set_theme(style="whitegrid", context="paper")

with PdfPages(pdf_filename) as pdf:
    
    # ---------------------------------------------------------
    # SLIDE 1: DATA
    # ---------------------------------------------------------
    fig1 = plt.figure(figsize=(12, 8.5))
    fig1.suptitle("Data", fontsize=18, fontweight='bold')
    
    # Top Left: Sample size, structure, and predictors
    ax1_tl = fig1.add_axes([0.05, 0.55, 0.4, 0.35])
    ax1_tl.axis('off')
    struct_txt = (
        f"DATASET STRUCTURE\n"
        f"----------------------------------------\n"
        f"Training Set : {len(train_df)} homes\n"
        f"Testing Set  : {len(test_df)} homes\n\n"
        f"TARGET VARIABLES\n"
        f"----------------------------------------\n"
        f"1. Sale Price (Continuous, kEUR)\n"
        f"2. Distressed Status (Binary 0/1)\n\n"
        f"PREDICTORS (12 Features)\n"
        f"----------------------------------------\n"
        f"{', '.join(feature_names)}"
    )
    ax1_tl.text(0, 1, struct_txt, fontsize=12, family='monospace', va='top', wrap=True)
    
    # Top Right: Summary Statistics
    ax1_tr = fig1.add_axes([0.55, 0.55, 0.4, 0.35])
    ax1_tr.axis('off')
    stats_txt = (
        f"SUMMARY STATISTICS (KEY FEATURES)\n"
        f"--------------------------------------------------\n"
        f"{train_df[['sale_price_keur', 'GrLivArea', 'OverallQual', 'YearBuilt']].describe().round(1).to_string()}"
    )
    ax1_tr.text(0, 1, stats_txt, fontsize=11, family='monospace', va='top')
    
    # Bottom Left: Distribution of Sale Prices
    ax1_bl = fig1.add_axes([0.05, 0.05, 0.4, 0.35])
    sns.histplot(train_df['sale_price_keur'], kde=True, color='dodgerblue', ax=ax1_bl)
    ax1_bl.set_title("Target 1: Distribution of Sale Prices (kEUR)", fontweight='bold')
    ax1_bl.set_xlabel("Sale Price")
    ax1_bl.set_ylabel("Count")
    
    # Bottom Right: Imbalance
    ax1_br = fig1.add_axes([0.55, 0.05, 0.4, 0.35])
    prevalence = train_df['distressed'].mean() * 100
    sns.countplot(x='distressed', data=train_df, palette=['lightgray', 'crimson'], ax=ax1_br)
    ax1_br.set_title(f"Target 2: Distressed Class Imbalance ({prevalence:.1f}%)", fontweight='bold')
    ax1_br.set_xlabel("Distressed Status")
    ax1_br.set_ylabel("Count")
    
    pdf.savefig(fig1)
    plt.close(fig1)

    # ---------------------------------------------------------
    # SLIDE 2: MODEL, METHOD AND RESULTS (Two-Column Layout)
    # ---------------------------------------------------------
    fig2 = plt.figure(figsize=(12, 8.5))
    fig2.suptitle("Model, Method and Results", fontsize=18, fontweight='bold')
    
    # Left Column: Formulas, Assumptions, Methodology
    ax2_left = fig2.add_axes([0.05, 0.05, 0.45, 0.85])
    ax2_left.axis('off')
    
    slide2_left_txt = (
        f"Model Formulas\n"
        f"--------------------------------------------------\n"
        f"1. Random Forest Regression / Classification Architecture:\n"
        f"   $f_{{RF}}(x) = \\frac{{1}}{{B}} \\sum_{{b=1}}^{{B}} T_b(x)$, where mtry = 4 randomly drawn\n\n"
        f"2. Gradient Boosting Optimization Equation:\n"
        f"   $F_M(x) = \\sum_{{m=1}}^{{M}} \\eta h_m(x)$, where $\\eta$ = {res['gbm_reg_hyperparams']['learning_rate']}\n\n\n"
        
        f"Model Assumptions\n"
        f"--------------------------------------------------\n"
        f"• Sampling Independence: Observations are assumed I.I.D.\n  Drawn from stable population.\n\n"
        f"• Invariant Split Monotonicity: Splits depend on ordinal\n  ranks. Completely immune to outliers.\n\n"
        f"• Imbalance Calibration: Assumes minority class requires\n  synthetic weights for cross-entropy.\n\n\n"
        
        f"Methodology & Libraries Used\n"
        f"--------------------------------------------------\n"
        f"• NumPy: Core CART Trees built strictly from scratch.\n"
        f"  Custom MDI Vector Logic: Programmed mathematical SSE tracking.\n\n"
        f"• Scikit-Learn: Executed benchmark architectures (GBM).\n"
        f"  Probability Calibration: Deployed balanced weights to\n  handle 15% class prevalence."
    )
    ax2_left.text(0, 0.95, slide2_left_txt, fontsize=12, family='monospace', va='top')

    # Right Column: Model Results
    ax2_right = fig2.add_axes([0.55, 0.05, 0.4, 0.85])
    ax2_right.axis('off')
    
    slide2_right_txt = (
        f"Model Results\n"
        f"--------------------------------------------------\n"
        f"• Task 1: Single CART Regressor (From Scratch)\n"
        f"  Target: sale_price_keur\n"
        f"  Training MSE: {mse_cart:,.2f}\n\n"
        
        f"• Task 2: Random Forest Regressor (From Scratch)\n"
        f"  Target: sale_price_keur\n"
        f"  Training MSE: {mse_rf:,.2f}\n\n"
        
        f"• Task 3: Gradient-Boosted Regressor (Library)\n"
        f"  Target: sale_price_keur\n"
        f"  Training MSE: {mse_gbm:,.2f} (Scikit-Learn)\n\n"
        
        f"• Task 4: Random Forest Classifier (Library)\n"
        f"  Target: distressed state\n"
        f"  Log-Loss: {loss_rf:.4f} (Balanced Weights)\n\n"
        
        f"• Task 5: Gradient-Boosted Classifier (Library)\n"
        f"  Target: distressed state\n"
        f"  Log-Loss: {loss_gbm:.4f} (Sample Weights)"
    )
    ax2_right.text(0, 0.95, slide2_right_txt, fontsize=12, family='monospace', va='top')

    pdf.savefig(fig2)
    plt.close(fig2)

    # ---------------------------------------------------------
    # SLIDE 3: PREDICTION COMPARISONS, TREES & IMPORTANCES
    # ---------------------------------------------------------
    fig3 = plt.figure(figsize=(12, 8.5))
    fig3.suptitle("Slide 3: Performance Comparisons & Architecture Visualization", fontsize=18, fontweight='bold')
    
    # Top Left: Regression Performance
    ax3_tl = fig3.add_axes([0.05, 0.55, 0.4, 0.35])
    sns.barplot(x=['CART Tree', 'Random Forest', 'Library GBM'], y=[mse_cart, mse_rf, mse_gbm], palette='Blues_r', ax=ax3_tl)
    ax3_tl.set_title("Regression Performance (Training MSE)", fontweight='bold')
    ax3_tl.set_ylabel("Mean Squared Error (Lower is Better)")
    for i, v in enumerate([mse_cart, mse_rf, mse_gbm]):
        ax3_tl.text(i, v + (v*0.02), f"{v:,.0f}", ha='center', fontweight='bold')

    # Top Right: Classification Performance
    ax3_tr = fig3.add_axes([0.55, 0.55, 0.4, 0.35])
    sns.barplot(x=['RF Classifier', 'GBM Classifier'], y=[loss_rf, loss_gbm], palette='Reds_r', ax=ax3_tr)
    ax3_tr.set_title("Classification Performance (Cross-Entropy Loss)", fontweight='bold')
    ax3_tr.set_ylabel("Log Loss (Lower is Better)")
    for i, v in enumerate([loss_rf, loss_gbm]):
        ax3_tr.text(i, v + 0.01, f"{v:.3f}", ha='center', fontweight='bold')

    # Bottom Left: Tree Visualization 
    ax3_bl = fig3.add_axes([0.05, 0.05, 0.45, 0.4])
    plot_tree(
        tree_reg, 
        feature_names=feature_names, 
        filled=True, 
        rounded=True, 
        ax=ax3_bl, 
        max_depth=2, 
        fontsize=6 # Slightly smaller to fit gracefully in half the slide
    )
    ax3_bl.set_title("CART Architecture (Extract of Top Nodes)", fontweight='bold')

    # Bottom Right: Feature Importances
    ax3_br = fig3.add_axes([0.55, 0.05, 0.4, 0.4])
    imp_reg = sorted(res['rf_reg_var_importance'].items(), key=lambda x: x[1], reverse=True)[:5]
    sns.barplot(x=[x[1] for x in imp_reg], y=[x[0] for x in imp_reg], ax=ax3_br, palette="viridis")
    ax3_br.set_title("Top 5 Drivers of Price (MDI Importances)", fontweight='bold')
    ax3_br.set_xlabel("Mean Decrease in Impurity")
    
    pdf.savefig(fig3)
    plt.close(fig3)


print(f"Success! {pdf_filename} has been saved.")