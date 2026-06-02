import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

#Load Dataset
df = pd.read_csv('hw3_data_s1467.csv')

#Basic Inspection
print("--- Dataset Info ---")
print(df.info())
print("\n--- First 5 Rows ---")
print(df.head())

#Statistical Summary
summary_stats = df.describe()
print("\n--- Summary Statistics ---")
print(summary_stats)

#Covariance Matrix
cov_matrix = df.cov()
print("\n--- Covariance Matrix ---")
print(cov_matrix)

#Visualization
plt.figure(figsize=(12, 5))

#Histogram
plt.subplot(1, 2, 1)
sns.histplot(df['asset1_return'], kde=True, color='skyblue', label='Asset 1 ($X$)', alpha=0.6)
sns.histplot(df['asset2_return'], kde=True, color='salmon', label='Asset 2 ($Y$)', alpha=0.6)
plt.title('Distribution of Asset Returns')
plt.xlabel('Daily Return')
plt.legend()

#Relationship between assets
plt.subplot(1, 2, 2)
sns.scatterplot(x='asset1_return', y='asset2_return', data=df, alpha=0.6)
plt.title('Asset 1 vs. Asset 2 Returns')
plt.xlabel('Asset 1 ($X$)')
plt.ylabel('Asset 2 ($Y$)')

#Add reference lines at zero
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.axvline(0, color='black', linestyle='--', linewidth=0.8)

plt.tight_layout()
plt.savefig('portfolio_eda_plots.png')
plt.show()