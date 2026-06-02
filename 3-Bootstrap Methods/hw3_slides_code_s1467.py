import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# Load data
df = pd.read_csv('hw3_data_s1467.csv')
n = len(df)
summary_stats = df.describe().round(6)

# Main Logic

def get_alpha(data):
    X, Y = data.iloc[:, 0].values, data.iloc[:, 1].values
    var_x, var_y = np.var(X, ddof=1), np.var(Y, ddof=1)
    cov_xy = np.cov(X, Y)[0, 1]
    return (var_y - cov_xy) / (var_x + var_y - 2 * cov_xy)

# Point Estimate
alpha_hat = get_alpha(df)

# Bootstrap - Explicit EDF Sampling (Mandatory Logic)
B = 1000
np.random.seed(42) 
boot_alphas = []
for _ in range(B):
    U = np.random.rand(n) 
    indices = np.floor(n * U).astype(int)
    boot_alphas.append(get_alpha(df.iloc[indices]))

boot_alphas = np.array(boot_alphas)
ci_lower, ci_upper = np.percentile(boot_alphas, [2.5, 97.5])

# Calculate Portfolio Variance Curve for Slide 3
var_x = np.var(df.iloc[:, 0], ddof=1)
var_y = np.var(df.iloc[:, 1], ddof=1)
cov_xy = np.cov(df.iloc[:, 0], df.iloc[:, 1])[0, 1]
alpha_range = np.linspace(0, 1, 100)
port_vars = (alpha_range**2 * var_x) + ((1 - alpha_range)**2 * var_y) + (2 * alpha_range * (1 - alpha_range) * cov_xy)

# PDF creation
with PdfPages('hw3_slides_s1467.pdf') as pdf:
    # ==========================================
    # SLIDE 1: DATA
    # ==========================================
    fig1, ax1 = plt.subplots(figsize=(11, 8.5))
    ax1.axis('off')

    #Title
    plt.suptitle('Data', fontsize=28, fontweight='bold', y=0.95)
    #text: sample size
    plt.text(0.05, 0.85, f"Sample Size: {n} observations", fontsize=16, transform=ax1.transAxes)

    # Summary Statistics Table
    # Creating a list of lists for the table content
    table_content = [['Metric', 'Asset 1 ($X$)', 'Asset 2 ($Y$)']]
    for idx, row in summary_stats.iterrows():
        table_content.append([idx.capitalize(), row['asset1_return'], row['asset2_return']])

    table = plt.table(cellText=table_content, 
                      loc='center left', 
                      bbox=[0.05, 0.4, 0.45, 0.35],
                      cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Histograms
    # Asset 1
    ax_h1 = fig1.add_axes([0.6, 0.55, 0.33, 0.25])
    sns.histplot(df['asset1_return'], kde=True, color='skyblue', ax=ax_h1)
    ax_h1.set_title('Histogram: Asset 1', fontsize=12)
    ax_h1.set_xlabel('Returns')

    # Asset 2
    ax_h2 = fig1.add_axes([0.6, 0.15, 0.33, 0.25])
    sns.histplot(df['asset2_return'], kde=True, color='salmon', ax=ax_h2)
    ax_h2.set_title('Histogram: Asset 2', fontsize=12)
    ax_h2.set_xlabel('Returns')

    # Save Slide
    pdf.savefig(fig1)
    plt.close(fig1)

    # ==========================================
    # SLIDE 2
    # ==========================================
    fig2, ax2 = plt.subplots(figsize=(11, 8.5))
    ax2.axis('off')

    plt.suptitle('Model, Method and Results', fontsize=28, fontweight='bold', y=0.95)

    # Formula
    plt.text(0.05, 0.82, '1. Model Formula:', fontsize=18, fontweight='bold', transform=ax2.transAxes)
    plt.text(0.1, 0.74, r'$\alpha = \frac{\sigma_Y^2 - \sigma_{XY}}{\sigma_X^2 + \sigma_Y^2 - 2\sigma_{XY}}$', 
             fontsize=26, transform=ax2.transAxes)
    
    # Method
    plt.text(0.05, 0.65, '2. Method:', fontsize=18, fontweight='bold', transform=ax2.transAxes)
    method_points = [
        "• Nonparametric Bootstrap (B=1000 resamples)",
        "• Explicit EDF Sampling: index = floor(n * U), U ~ Uniform(0,1)",
        "• Confidence Interval: Percentile Method (2.5%, 97.5%)",
        "• Libraries: pandas, numpy, matplotlib, seaborn"
    ]

    for i, point in enumerate(method_points):
        plt.text(0.07, 0.58 - (i*0.05), point, fontsize=14, transform=ax2.transAxes)

    # Assumptions
    plt.text(0.05, 0.35, '3. Model Assumptions:', fontsize=18, fontweight='bold', transform=ax2.transAxes)
    plt.text(0.07, 0.28, "• Returns are Independent and Identically Distributed (i.i.d).", 
             fontsize=14, transform=ax2.transAxes)
    plt.text(0.07, 0.23, "• Sample represents a stationary data-generating process.", 
             fontsize=14, transform=ax2.transAxes)
    
    # Results
    plt.text(0.05, 0.12, '4. Results:', fontsize=18, fontweight='bold', transform=ax2.transAxes)
    res_text = f"Alpha Hat: {alpha_hat:.4f}  |  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]"
    plt.text(0.07, 0.05, res_text, fontsize=18, fontweight='bold', color='navy', transform=ax2.transAxes)

    pdf.savefig(fig2)
    plt.close(fig2)

    # ==========================================
    # SLIDE 3
    # ==========================================
    fig3, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    plt.suptitle('Visualization', fontsize=28, fontweight='bold', y=0.95)

    # Plot A: Bootstrap Distribution of Alpha
    sns.histplot(boot_alphas, kde=True, color='purple', ax=axes[0, 0])
    axes[0, 0].axvline(alpha_hat, color='red', linestyle='--')
    axes[0, 0].set_title(r'Bootstrap $\hat{\alpha}$ Distribution')

    # Plot B: Empirical Distribution Function (EDF)
    x_sorted = np.sort(df.iloc[:, 0])
    y_edf = np.arange(1, n + 1) / n
    axes[0, 1].step(x_sorted, y_edf, where='post', color='darkgreen')
    axes[0, 1].set_title('Empirical Distribution (EDF)')

    # C. Scatter Plot (Asset 1 vs Asset 2)
    sns.regplot(x=df.iloc[:, 0], y=df.iloc[:, 1], ax=axes[1, 0], 
                scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    axes[1, 0].set_title('Asset Correlation (Scatter)')

    # D. Portfolio Variance Analysis (The "Impressive" V-Curve)
    axes[1, 1].plot(alpha_range, port_vars, color='darkblue', linewidth=2)
    axes[1, 1].axvline(alpha_hat, color='red', linestyle='--', alpha=0.7)
    axes[1, 1].scatter(alpha_hat, min(port_vars), color='red', zorder=5)
    axes[1, 1].set_title(r'Optimization: Total Variance vs $\alpha$')
    axes[1, 1].set_ylabel('Variance')

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    pdf.savefig(fig3)
    plt.close(fig3)