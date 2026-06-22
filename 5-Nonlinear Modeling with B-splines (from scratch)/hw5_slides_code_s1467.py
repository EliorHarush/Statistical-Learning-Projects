import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, log_loss, roc_auc_score, accuracy_score, f1_score

# =====================================================================
# 1. UTILITIES & B-SPLINE MATH 
# =====================================================================

def evaluate_bspline(x, degree, i, knots):
    if degree == 0:
        if knots[i] <= x < knots[i+1]:
            return 1.0
        elif knots[i] < knots[i+1] and x == knots[-1] and knots[i+1] == knots[-1]:
            return 1.0
        else:
            return 0.0
    denom1 = knots[i + degree] - knots[i]
    term1 = ((x - knots[i]) / denom1) * evaluate_bspline(x, degree - 1, i, knots) if denom1 > 0 else 0.0
    denom2 = knots[i + degree + 1] - knots[i + 1]
    term2 = ((knots[i + degree + 1] - x) / denom2) * evaluate_bspline(x, degree - 1, i + 1, knots) if denom2 > 0 else 0.0
    return term1 + term2

def create_knot_vector(degree, m_internal, t_min=0, t_max=40):
    lower_bound = [t_min] * (degree + 1)
    upper_bound = [t_max] * (degree + 1)
    internal_knots = list(np.linspace(t_min, t_max, m_internal + 2)[1:-1]) if m_internal > 0 else []
    return lower_bound + internal_knots + upper_bound

def build_design_matrix(T, H, W, degree, knots):
    n_samples = len(T)
    n_basis = len(knots) - degree - 1
    X = np.zeros((n_samples, n_basis + 2))
    for row in range(n_samples):
        temp = T.iloc[row] if isinstance(T, pd.Series) else T[row]
        for i in range(n_basis):
            X[row, i] = evaluate_bspline(temp, degree, i, knots)
    X[:, -2] = H  
    X[:, -1] = W  
    return X

def fit_ols(X, y):
    return np.linalg.solve(X.T @ X, X.T @ y)

# =========================================================
# 2. ACTIVE SLIDE GENERATION EXECUTION
# =========================================================

BEST_M = 3
print("--- Initializing Slide Generation ---")

train_df = pd.read_csv('hw5_train_s1467.csv')
test_df = pd.read_csv('hw5_test_s1467.csv')

with open('hw5_results_s1467.json', 'r') as f:
    results = json.load(f)

pred_consumption_kwh = results["pred_consumption_kwh"]
pred_high_demand_prob = results["pred_high_demand_prob"]

# Calculate equations and metrics
print("Calculating explicit model equations and performance metrics...")
final_knots = create_knot_vector(3, BEST_M, t_min=0, t_max=40)
X_train_full = build_design_matrix(train_df['temperature'], train_df['humidity'], train_df['weekend'], 3, final_knots)

# Equations (Added explicit line breaks to prevent overlapping text)
beta_reg = fit_ols(X_train_full, train_df['consumption_kwh'].values)
b_terms_reg = [f"{b:.2f}*B_{i+1}(T)" for i, b in enumerate(beta_reg[:-2])]
reg_eq = " + ".join(b_terms_reg[:3]) + "\n    + " + " + ".join(b_terms_reg[3:]) + f"\n    + {beta_reg[-2]:.2f}*H + {beta_reg[-1]:.2f}*W"

log_reg = LogisticRegression(fit_intercept=False, penalty=None, max_iter=1000).fit(X_train_full, train_df['high_demand_alert'].values)
b_terms_log = [f"{b:.2f}*B_{i+1}(T)" for i, b in enumerate(log_reg.coef_[0][:-2])]
log_eq = " + ".join(b_terms_log[:3]) + "\n    + " + " + ".join(b_terms_log[3:]) + f"\n    + {log_reg.coef_[0][-2]:.2f}*H + {log_reg.coef_[0][-1]:.2f}*W"

# Metrics
y_reg_pred = X_train_full @ beta_reg
reg_metrics = f"MSE: {mean_squared_error(train_df['consumption_kwh'], y_reg_pred):.3f}\nR²:  {r2_score(train_df['consumption_kwh'], y_reg_pred):.3f}"

log_probs = log_reg.predict_proba(X_train_full)[:, 1]
log_class = log_reg.predict(X_train_full)
class_metrics = f"Log Loss: {log_loss(train_df['high_demand_alert'], log_probs):.3f}\nAUC:      {roc_auc_score(train_df['high_demand_alert'], log_probs):.3f}\nAccuracy: {accuracy_score(train_df['high_demand_alert'], log_class):.3f}\nF1 Score: {f1_score(train_df['high_demand_alert'], log_class):.3f}"


# PDF Generation
plt.style.use('seaborn-v0_8-whitegrid')
pdf_filename = 'hw5_slides_s1467.pdf'

with PdfPages(pdf_filename) as pdf:
    
    # --- SLIDE 1: DATA ---
    fig1 = plt.figure(figsize=(16, 9))
    fig1.suptitle("Data Exploration", fontsize=24, fontweight='bold')
    
    gs1 = fig1.add_gridspec(2, 3)
    
    ax1 = fig1.add_subplot(gs1[0, 0])
    ax1.axis('off')
    text_structure = (
        "SAMPLE SIZE & STRUCTURE\n"
        "------------------------------------\n"
        f"• Train Set: {len(train_df)} independent days\n"
        f"• Test Set: {len(test_df)} independent days\n"
        "• Features: Temperature, Humidity, Weekend\n"
        "• Target 1 (Reg): Consumption (kWh)\n"
        "• Target 2 (Class): High Demand Alert (0/1)\n\n"
    )
    ax1.text(0.0, 0.5, text_structure, fontsize=13, va='center', ha='left')
    
    ax2 = fig1.add_subplot(gs1[1, 0])
    ax2.axis('off')
    text_stats = (
        "SUMMARY STATISTICS\n"
        "------------------------------------\n"
        f"• Temperature Range: {train_df['temperature'].min():.2f}°C to {train_df['temperature'].max():.2f}°C\n"
        "• Humidity Range: ~20% to ~90%\n"
        "• Highly non-linear 'U-Shape' behavior\n"
        "  observed in Temperature vs Consumption.\n"
        "• Demand Baseline Shift observed on weekends."
    )
    ax2.text(0.0, 0.5, text_stats, fontsize=13, va='center', ha='left')
    
    ax3 = fig1.add_subplot(gs1[0, 1])
    sns.scatterplot(data=train_df, x='temperature', y='consumption_kwh', hue='high_demand_alert', palette='coolwarm', ax=ax3)
    ax3.set_title("Temp vs Consumption (U-Shape)")
    
    ax4 = fig1.add_subplot(gs1[0, 2])
    sns.scatterplot(data=train_df, x='humidity', y='consumption_kwh', hue='high_demand_alert', palette='coolwarm', ax=ax4)
    ax4.set_title("Humidity vs Consumption")
    
    ax5 = fig1.add_subplot(gs1[1, 1])
    sns.boxplot(data=train_df, x='weekend', y='consumption_kwh', palette='Set2', ax=ax5)
    ax5.set_title("Weekend Effect")
    
    ax6 = fig1.add_subplot(gs1[1, 2])
    sns.histplot(data=train_df, x='temperature', hue='high_demand_alert', multiple='stack', palette='coolwarm', bins=20, ax=ax6)
    ax6.set_title("Alerts by Temperature Distribution")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    pdf.savefig(fig1)
    plt.close()

    # --- SLIDE 2: METHODOLOGY AND RESULTS (GRID LAYOUT) ---
    fig2 = plt.figure(figsize=(16, 9))
    fig2.suptitle("Methodology and Results", fontsize=24, fontweight='bold')
    gs2 = fig2.add_gridspec(2, 3)

    ax2_1 = fig2.add_subplot(gs2[0, :])
    ax2_1.axis('off')
    meth_text = (
        "METHODOLOGY:\n"
        "• Modeled non-linear temperature effects using Cubic B-Splines (Cox-de Boor algorithm, m=3 internal knots).\n"
        "• Task 1 mapped via custom Ordinary Least Squares (OLS). Task 2 mapped via Logistic Regression.\n\n"
        "ASSUMPTIONS:\n"
        "• Boundary knots clamped strictly at 0°C and 40°C to maintain stability at test-set extremes.\n"
        "• B-splines natively form a Partition of Unity (sum to 1), requiring the omission of a global intercept.\n"
        "• Humidity and Weekend status assumed as simple additive, linear effects."
    )
    ax2_1.text(0.05, 0.5, meth_text, fontsize=14, va='center', ha='left')

    ax2_2 = fig2.add_subplot(gs2[1, 0])
    ax2_2.axis('off')
    models_text = (
        "EXPLICIT MODELS:\n"
        "----------------------------\n\n"
        "Task 1 (Regression):\n"
        f"Y = {reg_eq}\n\n\n"
        "Task 2 (Classification):\n"
        "P(C=1) = 1 / (1 + exp(-Z))\n"
        f"Z = {log_eq}"
    )
    ax2_2.text(0.0, 0.5, models_text, fontsize=10, va='center', ha='left', family='monospace')

    ax2_3 = fig2.add_subplot(gs2[1, 1])
    ax2_3.axis('off')
    task1_perf_text = (
        "TASK 1 PERFORMANCE:\n"
        "--------------------------\n\n"
        f"{reg_metrics}\n\n\n"
    )
    ax2_3.text(0.05, 0.5, task1_perf_text, fontsize=11, va='center', ha='left', family='monospace')

    ax2_4 = fig2.add_subplot(gs2[1, 2])
    ax2_4.axis('off')
    task2_perf_text = (
        "TASK 2 PERFORMANCE:\n"
        "--------------------------\n\n"
        f"{class_metrics}\n\n"
    )
    ax2_4.text(0.05, 0.5, task2_perf_text, fontsize=11, va='center', ha='left', family='monospace')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    pdf.savefig(fig2)
    plt.close()
    
    # --- SLIDE 3: VISUALIZATION ---
    fig3 = plt.figure(figsize=(16, 9))
    fig3.suptitle("Visualization: Test Set Predictions & Model Behavior", fontsize=24, fontweight='bold')
    
    gs3 = fig3.add_gridspec(1, 2)
    
    ax_p1 = fig3.add_subplot(gs3[0, 0])
    ax_p1.scatter(train_df['temperature'], train_df['consumption_kwh'], color='lightgray', alpha=0.5, label='Train (True)')
    ax_p1.scatter(test_df['temperature'], pred_consumption_kwh, color='blue', alpha=0.8, marker='x', label='Test (Predicted)')
    ax_p1.set_title("Task 1: Predicted Consumption vs. Temperature", fontsize=14)
    ax_p1.set_xlabel("Temperature (°C)")
    ax_p1.set_ylabel("Consumption (kWh)")
    ax_p1.legend()
    
    ax_p2 = fig3.add_subplot(gs3[0, 1])
    colors = ['red' if p >= 0.5 else 'green' for p in pred_high_demand_prob]
    ax_p2.scatter(test_df['temperature'], pred_high_demand_prob, color=colors, alpha=0.7)
    ax_p2.axhline(0.5, linestyle='--', color='black', label='0.5 Alert Threshold')
    ax_p2.set_title("Task 2: Predicted High Demand Probability", fontsize=14)
    ax_p2.set_xlabel("Temperature (°C)")
    ax_p2.set_ylabel("Probability P(C_i=1)")
    ax_p2.legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    pdf.savefig(fig3)
    plt.close()

print(f"Success! Slides saved to {pdf_filename}")