import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load training data
df = pd.read_csv('hw4_train_s1467.csv')

# Separate predictors and target
X = df.drop(columns=['y'])
y = df['y']

# Setting up the plotting grid
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)

# --- Plot 1: Target Variable Distribution ---
ax1 = fig.add_subplot(gs[0, 0])
sns.histplot(y, kde=True, ax=ax1, color='navy')
ax1.set_title('Distribution of Target Variable (y)', fontweight='bold')
ax1.set_xlabel('y values')

# --- Plot 2: Predictor Scale/Variance Check ---
ax2 = fig.add_subplot(gs[0, 1])
std_devs = X.std()
std_devs.plot(kind='bar', ax=ax2, color='teal')
ax2.set_title('Standard Deviations of Predictors (x1-x40)', fontweight='bold')
ax2.set_xticks([])
ax2.set_xlabel('Predictors')
ax2.set_ylabel('Standard Deviation')

# --- Plot 3: Multicollinearity Heatmap ---
ax3 = fig.add_subplot(gs[1, 0])
# Calculate correlation matrix for predictors only
corr_matrix = X.corr()
# Create a mask to hide the upper triangle for readability
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0, 
            xticklabels=False, yticklabels=False, ax=ax3, cbar_kws={"shrink": .8})
ax3.set_title('Predictor Correlation Heatmap', fontweight='bold')

# --- Plot 4: Top 10 Correlated Features with 'y' ---
ax4 = fig.add_subplot(gs[1, 1])
corrs_with_y = df.corr()['y'].drop('y').abs().sort_values(ascending=False).head(10)
sns.barplot(x=corrs_with_y.values, y=corrs_with_y.index, ax=ax4, palette='viridis')
ax4.set_title('Top 10 Predictors Correlated with y (Absolute)', fontweight='bold')
ax4.set_xlabel('Absolute Correlation Coefficient')

plt.savefig('EDA plots.png', dpi=300, bbox_inches='tight')

plt.suptitle('EDA: High-Dimensional Regularization Dataset', fontsize=16, fontweight='bold', y=1.02)
plt.show()

print("--- Quick Scale Check ---")
print(f"Mean of means across all predictors: {X.mean().mean():.4f}")
print(f"Mean of std devs across all predictors: {X.std().mean():.4f}")