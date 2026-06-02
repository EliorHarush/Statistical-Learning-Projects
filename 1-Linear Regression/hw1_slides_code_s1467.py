import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

# ── Load data ──────────────────────────────────────────────────────────────
train = pd.read_csv(r"D:\לימודים\תואר שני\סמסטר ב\Statistical Learning\Projects\1.Linear Regression\Final\hw1_train_s1467.csv")
test = pd.read_csv(r"D:\לימודים\תואר שני\סמסטר ב\Statistical Learning\Projects\1.Linear Regression\Final\hw1_test_s1467.csv")

all_predictors = ["x1", "x2", "x3", "x4", "x5"]
X_train = train[all_predictors].values
y_train = train["y"].values
X_test = test[all_predictors].values
n_train, p = X_train.shape
n_test = len(X_test)

# ── Fit FULL OLS (all 5 predictors) ───────────────────────────────────────
X_train_i = np.column_stack([np.ones(n_train), X_train])
X_test_i = np.column_stack([np.ones(n_test), X_test])

beta_full = np.linalg.lstsq(X_train_i, y_train, rcond=None)[0]
y_train_pred_full = X_train_i @ beta_full
resid_full = y_train - y_train_pred_full
sigma_full = np.sqrt(np.sum(resid_full**2) / (n_train - p - 1))

# ── Coefficient inference (full model) ────────────────────────────────────
XtX_inv_full = np.linalg.inv(X_train_i.T @ X_train_i)
se_full = sigma_full * np.sqrt(np.diag(XtX_inv_full))
t_full = beta_full / se_full
p_full = 2 * (1 - stats.t.cdf(np.abs(t_full), df=n_train - p - 1))

coef_names_full = ["intercept"] + all_predictors

# ── Variable selection: backward elimination (p > 0.05) ──────────────────
ALPHA = 0.05
selected = [pred for i, pred in enumerate(all_predictors) if p_full[i+1] < ALPHA]
dropped = [pred for pred in all_predictors if pred not in selected]

# ── Fit REDUCED OLS (selected predictors only) ───────────────────────────
X_sel_train = train[selected].values
X_sel_test = test[selected].values
n_sel = len(selected)

X_sel_train_i = np.column_stack([np.ones(n_train), X_sel_train])
X_sel_test_i = np.column_stack([np.ones(n_test), X_sel_test])

beta_sel = np.linalg.lstsq(X_sel_train_i, y_train, rcond=None)[0]
y_train_pred_sel = X_sel_train_i @ beta_sel
resid_sel = y_train - y_train_pred_sel
sigma_sel = np.sqrt(np.sum(resid_sel**2) / (n_train - n_sel - 1))

coef_names_sel = ["intercept"] + selected

# ── Coefficient inference (reduced model) ─────────────────────────────────
XtX_inv_sel = np.linalg.inv(X_sel_train_i.T @ X_sel_train_i)
se_sel = sigma_sel * np.sqrt(np.diag(XtX_inv_sel))
t_sel = beta_sel / se_sel
p_sel = 2 * (1 - stats.t.cdf(np.abs(t_sel), df=n_train - n_sel - 1))

# ── Log-likelihood (reduced model) ───────────────────────────────────────
nll_per_obs = 0.5 * np.log(2 * np.pi * sigma_sel**2) + resid_sel**2 / (2 * sigma_sel**2)
nll_total = np.mean(nll_per_obs)

# ── ANOVA (reduced model) ────────────────────────────────────────────────
ss_res = np.sum(resid_sel**2)
ss_tot = np.sum((y_train - np.mean(y_train))**2)
ss_reg = ss_tot - ss_res
r_squared = 1 - ss_res / ss_tot
f_statistic = (ss_reg / n_sel) / (ss_res / (n_train - n_sel - 1))
f_pvalue = 1 - stats.f.cdf(f_statistic, n_sel, n_train - n_sel - 1)

# ── Correlation matrix ────────────────────────────────────────────────────
corr_matrix = train.corr()

# ══════════════════════════════════════════════════════════════════════════
# CREATE 3-SLIDE PDF
# ══════════════════════════════════════════════════════════════════════════

with PdfPages(r"D:\לימודים\תואר שני\סמסטר ב\Statistical Learning\Projects\1.Linear Regression\Final\hw1_slides_s1467.pdf") as pdf:

    # ══════════════════════════════════════════════════════════════════════
    # SLIDE 1: Data
    # ══════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(11, 8.5))
    gs = fig.add_gridspec(2, 12) # 12 is divisible by both 3 and 4
    fig.suptitle("Data", fontsize=18, fontweight='bold', y=0.98)
    
    ax_top = [fig.add_subplot(gs[0, i:i+4]) for i in range(0, 12, 4)]
    ax_bottom = [fig.add_subplot(gs[1, i:i+3]) for i in range(0, 12, 3)]

    # [MANDATORY] Sample size
    info_text = (
        f"Training observations: {n_train}\n"
        f"Test observations: {n_test}\n"
        f"Predictors: x1, x2, x3, x4, x5\n"
        f"Response: y (continuous)"
    )
    ax_top[0].text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center',fontfamily='monospace', transform=ax_top[0].transAxes)
    ax_top[0].set_title("Sample Size & Structure", fontsize=10, fontweight='bold')
    ax_top[0].axis('off')

    # [MANDATORY] Summary statistics
    summary_stats = train.describe().loc[['mean', 'std', 'min', 'max']]
    summary_text = summary_stats.to_string(float_format=lambda x: f"{x:.2f}")
    ax_top[1].text(0.05, 0.5, summary_text, fontsize=7, verticalalignment='center',
                     fontfamily='monospace', transform=ax_top[1].transAxes)
    ax_top[1].set_title("Summary Statistics", fontsize=10, fontweight='bold')
    ax_top[1].axis('off')

    # [OPTIONAL] Correlation heatmap
    im = ax_top[2].imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1)
    ax_top[2].set_xticks(range(len(corr_matrix.columns)))
    ax_top[2].set_yticks(range(len(corr_matrix.columns)))
    ax_top[2].set_xticklabels(corr_matrix.columns, fontsize=7, rotation=45)
    ax_top[2].set_yticklabels(corr_matrix.columns, fontsize=7)
    ax_top[2].set_title("Correlation Heatmap", fontsize=10, fontweight='bold')
    fig.colorbar(im, ax=ax_top[2], shrink=0.7)

    # [MANDATORY] Histograms of predictors and response
    for i, col in enumerate(["x1", "x3", "x5", "y"]):
        ax = ax_bottom[i]
        ax.hist(train[col], bins=25, color='steelblue', edgecolor='white', alpha=0.8)
        ax.set_title(f"Histogram of {col}", fontsize=10, fontweight='bold')
        ax.set_xlabel(col, fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════
    # SLIDE 2: Model, Method and Results (text/tables only — no plots)
    #
    # Layout (logical flow left → right):
    #   Top row:    Model & Assumptions  |  Method (library)        |  Variable Selection
    #   Bottom row: Reduced Model Formula|  Coefficients (t-test)   |  ANOVA, NLL & Summary
    # ══════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 3, figsize=(11, 8.5))
    fig.suptitle("Model, Method and Results", fontsize=18, fontweight='bold', y=0.98)

    # ── TOP-LEFT [0,0]: Model & Assumptions ──────────────────────────────
    # [MANDATORY] Model formula + [MANDATORY] Assumptions
    model_text = (
        "Model:\n"
        "  y = beta_0 + beta_1*x1 + ... + beta_p*xp\n"
        "      + epsilon,   epsilon ~ N(0, sigma^2)\n\n"
        "Assumptions:\n"
        "  1. Linearity: y is a linear function of x\n"
        "  2. Independence: observations are i.i.d.\n"
        "  3. Normality: residuals ~ Normal\n"
        "  4. Homoscedasticity: constant variance\n"
        "     (equal variance of errors)"
    )
    axes[0, 0].text(0.05, 0.5, model_text, fontsize=8.5, verticalalignment='center',
                     fontfamily='monospace', transform=axes[0, 0].transAxes)
    axes[0, 0].set_title("Model & Assumptions", fontsize=10, fontweight='bold')
    axes[0, 0].axis('off')

    # ── TOP-CENTER [0,1]: Method (library/package) ───────────────────────
    # [MANDATORY] Library/package used
    method_text = (
        "Method:\n"
        "  Ordinary Least Squares (OLS)\n"
        "  via normal equations\n\n"
        "Library / Package:\n"
        "  numpy  — OLS fitting (lstsq)\n"
        "  scipy.stats — t-test, F-test\n"
        "  pandas — data handling\n\n"
        "Variable Selection Strategy:\n"
        "  Backward elimination using\n"
        "  individual p-values (alpha=0.05)"
    )
    axes[0, 1].text(0.05, 0.5, method_text, fontsize=8.5, verticalalignment='center',
                     fontfamily='monospace', transform=axes[0, 1].transAxes)
    axes[0, 1].set_title("Method", fontsize=10, fontweight='bold')
    axes[0, 1].axis('off')

    # ── TOP-RIGHT [0,2]: Variable Selection ──────────────────────────────
    # [OPTIONAL] Variable selection details
    sel_text = (
        "Variable Selection\n"
        "(backward elimination, alpha=0.05)\n\n"
        "Full model p-values:\n"
    )
    for i, name in enumerate(coef_names_full):
        sig = "***" if p_full[i] < 0.001 else "**" if p_full[i] < 0.01 else "*" if p_full[i] < 0.05 else ""
        mark = " <-- drop" if i > 0 and p_full[i] >= ALPHA else ""
        sel_text += f"  {name:>10s}: p={p_full[i]:.4f} {sig}{mark}\n"
    sel_text += f"\nSelected: {', '.join(selected)}\n"
    sel_text += f"Dropped:  {', '.join(dropped)}"

    axes[0, 2].text(0.05, 0.5, sel_text, fontsize=7.5, verticalalignment='center',
                     fontfamily='monospace', transform=axes[0, 2].transAxes)
    axes[0, 2].set_title("Variable Selection", fontsize=10, fontweight='bold')
    axes[0, 2].axis('off')

    # ── BOTTOM-LEFT [1,0]: Reduced Model Formula ─────────────────────────
    # [MANDATORY] Model formula with estimated quantities
    formula_text = (
        "Reduced Model (after selection):\n\n"
        f"  y = beta_0 + " +
        " + ".join(f"beta_{selected[i][1:]}*{selected[i]}" for i in range(n_sel)) +
        " + epsilon\n\n"
        "Estimated coefficients:\n"
    )
    for i, name in enumerate(coef_names_sel):
        formula_text += f"  {name:>10s} = {beta_sel[i]:>8.4f}  (SE={se_sel[i]:.4f})\n"
    formula_text += f"\n  sigma_hat = {sigma_sel:.4f}"

    axes[1, 0].text(0.05, 0.5, formula_text, fontsize=8, verticalalignment='center',
                     fontfamily='monospace', transform=axes[1, 0].transAxes)
    axes[1, 0].set_title("Reduced Model Formula", fontsize=10, fontweight='bold')
    axes[1, 0].axis('off')

    # ── BOTTOM-CENTER [1,1]: Coefficient table ───────────────────────────
    # [OPTIONAL] Coefficient table with t-test and p-values (reduced model)
    table_text = f"{'Name':>10s}  {'Coef':>8s}  {'SE':>7s}  {'t':>7s}  {'p-value':>8s}\n"
    table_text += "-" * 50 + "\n"
    for i, name in enumerate(coef_names_sel):
        sig = "***" if p_sel[i] < 0.001 else "**" if p_sel[i] < 0.01 else "*" if p_sel[i] < 0.05 else ""
        table_text += f"{name:>10s}  {beta_sel[i]:>8.4f}  {se_sel[i]:>7.4f}  {t_sel[i]:>7.2f}  {p_sel[i]:>8.4f} {sig}\n"
    axes[1, 1].text(0.05, 0.5, table_text, fontsize=7.5, verticalalignment='center',
                     fontfamily='monospace', transform=axes[1, 1].transAxes)
    axes[1, 1].set_title("Coefficients (t-test & p-values)", fontsize=10, fontweight='bold')
    axes[1, 1].axis('off')

    # ── BOTTOM-RIGHT [1,2]: ANOVA, NLL & Summary ─────────────────────────
    # [OPTIONAL] ANOVA + [OPTIONAL] NLL
    summary_text = (
        f"ANOVA (reduced model)\n"
        f"  F = {f_statistic:.2f}, p = {f_pvalue:.2e}\n"
        f"  R-squared: {r_squared:.4f}\n"
        f"  Adj R-sq:  {1 - (1-r_squared)*(n_train-1)/(n_train-n_sel-1):.4f}\n\n"
        f"Negative Log-Likelihood (NLL):\n"
        f"  NLL_i = 0.5*log(2*pi*sigma^2)\n"
        f"        + (y_i - mu_i)^2/(2*sigma^2)\n"
        f"  Mean NLL (train) = {nll_total:.4f}\n\n"
        f"sigma_hat: {sigma_sel:.4f}\n"
        f"Selected:  {', '.join(selected)}\n"
        f"Dropped:   {', '.join(dropped)}"
    )
    axes[1, 2].text(0.05, 0.5, summary_text, fontsize=8, verticalalignment='center',
                     fontfamily='monospace', transform=axes[1, 2].transAxes)
    axes[1, 2].set_title("ANOVA, NLL & Summary", fontsize=10, fontweight='bold')
    axes[1, 2].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════
    # SLIDE 3: Visualization (all plots go here)
    # ══════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 3, figsize=(11, 8.5))
    fig.suptitle("Visualization", fontsize=18, fontweight='bold', y=0.98)

    # [OPTIONAL] Residuals vs fitted (reduced model)
    axes[0, 0].scatter(y_train_pred_sel, resid_sel, s=12, alpha=0.5, color='steelblue')
    axes[0, 0].axhline(y=0, color='red', linewidth=1, linestyle='--')
    axes[0, 0].set_xlabel("Fitted Values", fontsize=9)
    axes[0, 0].set_ylabel("Residuals", fontsize=9)
    axes[0, 0].set_title("Residuals vs Fitted", fontsize=10, fontweight='bold')

    # [OPTIONAL] QQ plot of residuals
    stats.probplot(resid_sel, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title("QQ Plot of Residuals", fontsize=10, fontweight='bold')

    # [OPTIONAL] Predicted vs actual (reduced model)
    axes[0, 2].scatter(y_train, y_train_pred_sel, s=12, alpha=0.5, color='steelblue')
    lims = [min(y_train.min(), y_train_pred_sel.min()),
            max(y_train.max(), y_train_pred_sel.max())]
    axes[0, 2].plot(lims, lims, 'r--', linewidth=1)
    axes[0, 2].set_xlabel("Actual y", fontsize=9)
    axes[0, 2].set_ylabel("Predicted y", fontsize=9)
    axes[0, 2].set_title("Predicted vs Actual", fontsize=10, fontweight='bold')

    # [OPTIONAL] Scatter plot: y vs x3 (strongest predictor)
    axes[1, 0].scatter(train["x3"], train["y"], s=15, alpha=0.6, color='steelblue')
    axes[1, 0].set_xlabel("x3", fontsize=9)
    axes[1, 0].set_ylabel("y", fontsize=9)
    axes[1, 0].set_title("Scatter: y vs x3", fontsize=10, fontweight='bold')

    # [OPTIONAL] Box plots of predictors
    train[all_predictors].boxplot(ax=axes[1, 1])
    axes[1, 1].set_title("Box Plots of Predictors", fontsize=10, fontweight='bold')
    axes[1, 1].set_ylabel("Value", fontsize=9)

    # [OPTIONAL] Histogram of residuals
    axes[1, 2].hist(resid_sel, bins=25, color='steelblue', edgecolor='white', alpha=0.8)
    axes[1, 2].set_xlabel("Residual", fontsize=9)
    axes[1, 2].set_ylabel("Count", fontsize=9)
    axes[1, 2].set_title("Histogram of Residuals", fontsize=10, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)

print("Saved hw1_slides_s0000.pdf (3 slides)")