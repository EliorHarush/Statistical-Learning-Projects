import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the Data
train_df = pd.read_csv('hw6_train_s1467.csv')

print("=== Dataset Info ===")
train_df.info()

print("\n=== Summary Statistics ===")
print(train_df.describe())

# 2. Analyze the Outcomes (Targets)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Target 1: Sale Price (Regression)
sns.histplot(train_df['sale_price_keur'], kde=True, ax=axes[0], color='blue')
axes[0].set_title('Distribution of Sale Price (sale_price_keur)')
axes[0].set_xlabel('Price (kEUR)')
axes[0].set_ylabel('Frequency')

# Target 2: Distressed Flag (Classification)
sns.countplot(data=train_df, x='distressed', ax=axes[1], palette='Set2')
axes[1].set_title('Class Imbalance: Distressed Sales')
axes[1].set_xlabel('Distressed (0 = No, 1 = Yes)')
axes[1].set_ylabel('Count')

# Calculate the exact prevalence
prevalence = train_df['distressed'].mean()
print(f"\nPrevalence of 'distressed' class: {prevalence:.2%}")
plt.tight_layout()
plt.show()

# 3. Correlation Matrix (Linear Relationships)
# Note: Trees don't require linear relationships, but this helps us spot redundancy 
# and strong univariate predictors.
plt.figure(figsize=(12, 8))
corr_matrix = train_df.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Feature Correlation Matrix')
plt.show()

# 4. Bivariate Analysis: Features vs Sale Price
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# GrLivArea vs Sale Price
sns.scatterplot(data=train_df, x='GrLivArea', y='sale_price_keur', hue='distressed', alpha=0.7, ax=axes[0])
axes[0].set_title('Living Area vs Sale Price')

# OverallQual vs Sale Price (Boxplot for ordinal)
sns.boxplot(data=train_df, x='OverallQual', y='sale_price_keur', ax=axes[1], palette='viridis')
axes[1].set_title('Overall Quality vs Sale Price')

plt.tight_layout()
plt.show()

# 5. Feature Distributions Split by Distressed Class
# This helps us see if certain features are highly predictive of a distressed sale
features_to_plot = ['YearBuilt', 'OverallCond', 'GrLivArea', 'sale_price_keur']
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, feature in enumerate(features_to_plot):
    sns.kdeplot(data=train_df, x=feature, hue='distressed', fill=True, common_norm=False, ax=axes[i], alpha=0.5)
    axes[i].set_title(f'{feature} Distribution by Distressed')

plt.tight_layout()
plt.show()