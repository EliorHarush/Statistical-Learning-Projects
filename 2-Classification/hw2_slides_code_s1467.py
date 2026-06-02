
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, log_loss
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# -- Load Data --
train_df = pd.read_csv('hw2_train_s1467.csv')
test_df = pd.read_csv('hw2_test_s1467.csv')

train_df['x1x3'] = train_df['x1'] * train_df['x3']
test_df['x1x3'] = test_df['x1'] * test_df['x3']

n_train = len(train_df)
n_test = len(test_df)

# ── Data Exploration ──────────────────────────────────────────────────────────────

# -- Basic Overview --
#print(train_df.info())
#print(train_df.describe())

# -- Check Class Balance --
#print(train_df['y'].value_counts(normalize=True))

# -- Set Visual Style --
sns.set_theme(style='whitegrid')
features = ['x1', 'x2', 'x3']
corr_matrix = train_df[['x1', 'x2', 'x3', 'x1x3', 'y']].corr()
fig,axes = plt.subplots(1, 3, figsize=(15, 5))

for i, col in enumerate(features):
    sns.boxplot(data=train_df, x='y', y=col, ax=axes[i])
    axes[i].set_title(f'Distribution of {col} by Class')

plt.tight_layout() ## Box plot for x2 shows stronger univariate signal

# -- Identifying Non-Linearity --
sns.pairplot(train_df, vars=['x1', 'x2', 'x3',], hue='y', markers=["o", "s"], diag_kind='kde')
plt.suptitle("Pairwise Relationships and Class Separation", y=1.02) ## getting hints of correlation between x1 and x3

# -- Correlation Analysis --
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', fmt=".2f")
plt.title("Feature Correlation Matrix")

#plt.show()

# ── Model Fitting ────────────────────────────────────────────────────────
# Prepare features for Statsmodels (including the constant intercept)
X_features = train_df[['x1', 'x2', 'x3', 'x1x3']]
X_with_const = sm.add_constant(X_features)
y_target = train_df['y']

# Fit the Logit model (MLE)
logit_model = sm.Logit(y_target, X_with_const).fit(disp=0)

# Calculate validation metrics for the summary section
y_prob = logit_model.predict(X_with_const)
y_pred = (y_prob > 0.5).astype(int)
cm = confusion_matrix(y_target, y_pred)
final_accuracy = (y_pred == y_target).mean()
final_log_loss = log_loss(y_target, y_prob)

# ── QDA Fitting for Comparison ──────────────────────────────────────────
qda = QuadraticDiscriminantAnalysis()
qda.fit(train_df[['x1', 'x2', 'x3']], train_df['y'])
y_prob_qda = qda.predict_proba(train_df[['x1', 'x2', 'x3']])[:, 1]
y_pred_qda = qda.predict(train_df[['x1', 'x2', 'x3']])
qda_acc = (y_pred_qda == train_df['y']).mean()

# ── Slide Generation ────────────────────────────────────────────────────────

with PdfPages('hw2_slides_s1467.pdf') as pdf:
    # --- SLIDE 1: DATA ---
    fig = plt.figure(figsize=(12, 8.5))
    # Using 15 units: Top (5+5+5) and Bottom (3+3+3+3+3)
    gs = fig.add_gridspec(2, 15, hspace=0.4, wspace=0.6) 
    fig.suptitle("Data", fontsize=18, fontweight='bold', y=0.98)
    
    # Top Row: 3 Slots (Sample Size, Stats, Heatmap)
    ax_info = fig.add_subplot(gs[0, 0:5])
    ax_stats = fig.add_subplot(gs[0, 5:10])
    ax_heat = fig.add_subplot(gs[0, 10:15])

    # Bottom Row: 5 Slots (x1, x2, x3, y, Box Plot)
    ax_b0 = fig.add_subplot(gs[1, 0:3])
    ax_b1 = fig.add_subplot(gs[1, 3:6])
    ax_b2 = fig.add_subplot(gs[1, 6:9])
    ax_b3 = fig.add_subplot(gs[1, 9:12])
    ax_box = fig.add_subplot(gs[1, 12:15])
    ax_bottom = [ax_b0, ax_b1, ax_b2, ax_b3]

    # [1.1] Sample Size & Structure
    info_text = (
        f"Training observations: {n_train}\n"
        f"Test observations: {n_test}\n"
        f"Predictors: x1, x2, x3, x1x3\n"
        f"Response: y (Binary Classification)\n"
        f"Class 1 Balance: {train_df.y.mean():.1%}"
    )
    ax_info.text(0.05, 0.5, info_text, fontsize=9.5, verticalalignment='center',
                  fontfamily='monospace', transform=ax_info.transAxes)
    ax_info.set_title("Sample Size & Structure", fontsize=10, fontweight='bold')
    ax_info.axis('off')

    # [1.2] Summary Statistics
    summary_stats = train_df[['x1', 'x2', 'x3', 'y']].describe().loc[['mean', 'std', 'min', 'max']]
    summary_text = summary_stats.to_string(float_format=lambda x: f"{x:.2f}")
    ax_stats.text(0.0, 0.5, summary_text, fontsize=7.5, verticalalignment='center',
                  fontfamily='monospace', transform=ax_stats.transAxes)
    ax_stats.set_title("Summary Statistics", fontsize=10, fontweight='bold')
    ax_stats.axis('off')

    # [1.3] Correlation Heatmap (FIXED: grid=False)
    im = ax_heat.imshow(corr_matrix.values, cmap='coolwarm', vmin=-1, vmax=1)
    ax_heat.grid(False) 
    ax_heat.set_xticks(range(len(corr_matrix.columns)))
    ax_heat.set_yticks(range(len(corr_matrix.columns)))
    ax_heat.set_xticklabels(corr_matrix.columns, fontsize=7, rotation=45)
    ax_heat.set_yticklabels(corr_matrix.columns, fontsize=7)
    ax_heat.set_title("Interaction Heatmap", fontsize=10, fontweight='bold')
    fig.colorbar(im, ax=ax_heat, shrink=0.7)

    # [1.4] Bottom Row: Histograms of x1, x2, x3, and y
    plot_cols = ["x1", "x2", "x3", "y"]
    for i, col in enumerate(plot_cols):
        ax = ax_bottom[i]
        ax.hist(train_df[col], bins=20, color='steelblue', edgecolor='white', alpha=0.8)
        ax.set_title(f"Histogram of {col}", fontsize=9, fontweight='bold')
        ax.set_xlabel(col, fontsize=8)
        ax.tick_params(labelsize=7)

    # [1.5] Bottom Row: Grouped Box Plot (Now in its own slot!)
    # Visual proof that x2 is a strong linear predictor
    train_melt = train_df.melt(id_vars='y', value_vars=['x1', 'x2', 'x3'])
    sns.boxplot(data=train_melt, x='variable', y='value', hue='y', 
                ax=ax_box, palette='coolwarm', fliersize=1)
    ax_box.set_title("Predictors vs Class (y)", fontsize=9, fontweight='bold')
    ax_box.set_xlabel("Predictor", fontsize=8)
    ax_box.set_ylabel("Value", fontsize=8)
    ax_box.legend(title='y', fontsize=6, title_fontsize=7, loc='upper right')
    ax_box.tick_params(labelsize=7)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)

    # --- SLIDE 2: METHOD AND RESULTS ---
    fig, axes = plt.subplots(2, 3, figsize=(11, 8.5))
    fig.suptitle("Model, Method and Results", fontsize=18, fontweight='bold', y=0.98)
    
    model_assumptions = (
        "Logistic Model:\n"
        "  Logit(p) = b0 + b1*x1 + b2*x2\n"
        "             + b3*x3 + b4*(x1*x3)\n"
        "  where p = P(y=1)\n\n"
        "Assumptions:\n"
        "  1. Linearity of the log-odds\n"
        "  2. Independence of observations\n"
        "  3. Lack of Multicollinearity\n"
        "  4. Proper specification (Interaction)"
    )
    axes[0, 0].text(0.05, 0.5, model_assumptions, fontsize=8.5, verticalalignment='center',
                 fontfamily='monospace', transform=axes[0, 0].transAxes)
    axes[0, 0].set_title("Model & Assumptions", fontsize=10, fontweight='bold')
    axes[0, 0].axis('off')

    method_text = (
        "Method:\n"
    "  Logistic Regression\n"
    "  Maximum Likelihood Est. (MLE)\n"
    "  QDA\n\n "
    "Library / Package:\n"
    "  statsmodels (sm.Logit)\n"
    "  sklearn\n"
    "  pandas & numpy (data handling)\n\n"
    )
    axes[0, 1].text(0.05, 0.5, method_text, fontsize=8.5, verticalalignment='center',
                 fontfamily='monospace', transform=axes[0, 1].transAxes)
    axes[0, 1].set_title("Method", fontsize=10, fontweight='bold')
    axes[0, 1].axis('off')

    logic_text = (
        "Strategy:\n"
    "  - x2: Retained as primary linear driver.\n"
    "  - x1 & x3: Included to capture the\n"
    "    XOR-style interaction term.\n\n"
    "Marginality Principle:\n"
    "  Main effects (x1, x3) were kept in\n"
    "  the model to support the higher-order\n"
    "  interaction term (x1*x3)."
    )
    axes[0, 2].text(0.05, 0.5, logic_text, fontsize=8.5, verticalalignment='center',
                 fontfamily='monospace', transform=axes[0, 2].transAxes)
    axes[0, 2].set_title("Logic & Strategy", fontsize=10, fontweight='bold')
    axes[0, 2].axis('off')

    params = logit_model.params
    formula_text = (
        "Estimated Equation (Log-Odds):\n"
    f"  z = {params[0]:.3f} + {params[1]:.3f}*x1\n"
    f"      + {params[2]:.3f}*x2 + {params[3]:.3f}*x3\n"
    f"      + {params[4]:.3f}*(x1*x3)\n\n"
    "Classification Rule:\n"
    "  y_hat = 1 if [1 / (1 + exp(-z))] > 0.5\n"
    "  else 0"
    )
    axes[1, 0].text(0.05, 0.5, formula_text, fontsize=8, verticalalignment='center',
                 fontfamily='monospace', transform=axes[1, 0].transAxes)
    axes[1, 0].set_title("Model Formula", fontsize=10, fontweight='bold')
    axes[1, 0].axis('off')

    p_vals = logit_model.pvalues
    se = logit_model.bse
    names = ["const", "x1", "x2", "x3", "x1*x3"]

    table_header = f"{'Name':<7} {'Coef':>7} {'SE':>6} {'p':>6}\n"
    table_header += "-" * 32 + "\n"
    table_rows = ""
    for i, name in enumerate(names):
        sig = "***" if p_vals[i] < 0.001 else "**" if p_vals[i] < 0.01 else "*" if p_vals[i] < 0.05 else ""
        table_rows += f"{name:<7} {params[i]:>7.3f} {se[i]:>6.3f} {p_vals[i]:>6.3f}{sig}\n"

    axes[1, 1].text(0.05, 0.5, table_header + table_rows, fontsize=8, verticalalignment='center',
                 fontfamily='monospace', transform=axes[1, 1].transAxes)
    axes[1, 1].set_title("Coefficients (z-test)", fontsize=10, fontweight='bold')
    axes[1, 1].axis('off')

    perf_text = (
        f"Performance Summary:\n"
        f"  Accuracy: {final_accuracy:.2%}\n"
        f"  Log-Loss: {final_log_loss:.4f}\n"
        f"  Pseudo R2: {logit_model.prsquared:.4f}\n\n"
        f"Optimization:\n"
        f"  Converged: {logit_model.mle_retvals['converged']}\n"
        f"  Iterations: {logit_model.mle_retvals['iterations']}\n"
        f"  LLR p-value: {logit_model.llr_pvalue:.2e}"
    )
    axes[1, 2].text(0.05, 0.5, perf_text, fontsize=8.5, verticalalignment='center',
                 fontfamily='monospace', transform=axes[1, 2].transAxes)
    axes[1, 2].set_title("Performance Summary", fontsize=10, fontweight='bold')
    axes[1, 2].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)

    # --- Slide 3: Visualization ---
    fig, axes = plt.subplots(2, 3, figsize=(11, 8.5))
    fig.suptitle("Visualization & Comparison", fontsize=18, fontweight='bold', y=0.98)
    
    # ROC Curve Comparison (Logistic vs QDA)
    fpr_logit, tpr_logit, _ = roc_curve(train_df['y'], y_prob)
    roc_auc_logit = auc(fpr_logit, tpr_logit)
    
    fpr_qda, tpr_qda, _ = roc_curve(train_df['y'], y_prob_qda)
    roc_auc_qda = auc(fpr_qda, tpr_qda)
    
    axes[0, 0].plot(fpr_logit, tpr_logit, color='darkorange', lw=2, label=f'Logit (AUC={roc_auc_logit:.2f})')
    axes[0, 0].plot(fpr_qda, tpr_qda, color='seagreen', lw=2, label=f'QDA (AUC={roc_auc_qda:.2f})')
    axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    axes[0, 0].set_title("ROC Comparison", fontsize=10, fontweight='bold')
    axes[0, 0].legend(fontsize=7, loc="lower right")

    # Logit Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1], cbar=False)
    axes[0, 1].set_title("Logit Confusion Matrix", fontsize=10, fontweight='bold')
    axes[0, 1].set_xlabel("Predicted")
    axes[0, 1].set_ylabel("Actual")

    # Interaction Plot: x1 vs x3
    # This visually proves the XOR pattern we discussed
    axes[0, 2].axis('off')
    inner_gs = axes[0, 2].get_subplotspec().subgridspec(3, 3, wspace=0.1, hspace=0.1)
    pair_vars = ['x1', 'x2', 'x3']

    for i in range(3):
        for j in range(3):
            inner_ax = fig.add_subplot(inner_gs[i, j])
            if i == j:
                # Diagonal: Histograms
                inner_ax.hist(train_df[pair_vars[i]], bins=15, color='gray', alpha=0.6)
            else:
            # Off-diagonal: Scatter plots colored by class y
                inner_ax.scatter(train_df[pair_vars[j]], train_df[pair_vars[i]], 
                             c=train_df['y'], cmap='coolwarm', s=1, alpha=0.4)
        
             # Clean up the mini-plots
            inner_ax.set_xticks([]); inner_ax.set_yticks([])
            if i == 2: inner_ax.set_xlabel(pair_vars[j], fontsize=6)
            if j == 0: inner_ax.set_ylabel(pair_vars[i], fontsize=6)

    axes[0, 2].set_title("Pairplot: x1, x2, x3", fontsize=10, fontweight='bold', pad=15)

    # Histogram of Predicted Probabilities (Logit)
    axes[1, 0].hist(y_prob, bins=20, color='steelblue', edgecolor='white', alpha=0.8)
    axes[1, 0].set_title("Logit Probability Dist.", fontsize=10, fontweight='bold')
    axes[1, 0].set_xlabel("Predicted Probability")

    # Predictor Box Plots (Same style as reference)
    train_df[['x1', 'x2', 'x3']].boxplot(ax=axes[1, 1])
    axes[1, 1].set_title("Predictor Box Plots", fontsize=10, fontweight='bold')

    # Direct Accuracy Comparison
    axes[1, 2].bar(['Logit', 'QDA'], [final_accuracy, qda_acc], color=['darkorange', 'seagreen'])
    axes[1, 2].set_title("Method Accuracy Comparison", fontsize=10, fontweight='bold')
    axes[1, 2].set_ylim(0, 1)
    # Add labels to bars
    axes[1, 2].text(0, final_accuracy + 0.02, f'{final_accuracy:.2%}', ha='center', fontsize=8)
    axes[1, 2].text(1, qda_acc + 0.02, f'{qda_acc:.2%}', ha='center', fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)

print("Analysis complete: 'hw2_results_s1467.pdf' and 'hw2_slides_s1467.pdf' created.")

