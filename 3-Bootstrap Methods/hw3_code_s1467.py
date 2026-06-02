import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

#Load the data
df = pd.read_csv('hw3_data_s1467.csv')
n = len(df)

def get_alpha(data):
    #Calculates optimal alpha for a provided dataframe.

    X = data.iloc[:, 0].values
    Y = data.iloc[:, 1].values

    #Calculate means
    mu_x, mu_y = np.mean(X), np.mean(Y)

    #Calculate variances and covariances
    n = len(data)
    var_x = np.sum((X-mu_x)**2) / (n-1)
    var_y = np.sum((Y - mu_y)**2) / (n - 1)
    cov_xy = np.sum((X - mu_x) * (Y - mu_y)) / (n - 1)

    #Apply formula
    numerator = var_y - cov_xy
    denominator = var_x + var_y - 2 * cov_xy

    return numerator / denominator

# --- BOOTSTRAP FROM SCRATCH (EXPLICIT EDF METHOD) ---
B = 1000
bootstrap_alphas = []

np.random.seed(42) # For reproducibility

for i in range(B):
    # generate uniform random distribution
    U = np.random.rand(n)

    # convert to random indices
    indices = np.floor(n * U).astype(int)

    # resample rows and compute alpha
    boot_sample = df.iloc[indices]
    bootstrap_alphas.append(get_alpha(boot_sample))

bootstrap_alphas = np.array(bootstrap_alphas)

# --- RESULTS & PERCENTILE CI ---
alpha_hat = get_alpha(df)
ci_lower = np.percentile(bootstrap_alphas, 2.5)
ci_upper = np.percentile(bootstrap_alphas, 97.5)

print(f"Alpha Hat: {alpha_hat:.4f}")
print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

# Save to CSV
results_df = pd.DataFrame({
    'alpha_hat': [alpha_hat],
    'ci_lower': [ci_lower],
    'ci_upper': [ci_upper]
})
results_df.to_csv('hw3_bootstrap_s1467.csv', index=False)

