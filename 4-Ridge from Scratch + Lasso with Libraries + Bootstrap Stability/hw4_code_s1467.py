import numpy as np
import pandas as pd
import json
from sklearn.linear_model import LassoCV

# ==========================================
# 1. DATA LOADING & PREPARATION
# ==========================================

train_df = pd.read_csv('hw4_train_s1467.csv')
test_df = pd.read_csv('hw4_test_s1467.csv')
lambda_grid = pd.read_csv('hw4_lambda_grid_s1467.csv').values.flatten()

X_train_full = train_df.drop(columns=['y']).values
y_train_full = train_df['y'].values

X_test = test_df.values
# ==========================================
# TASK 1 FUNCTIONS: Ridge Regression
# ==========================================

# ==========================================
# HELPER: Standardization
# ==========================================
def standardize(X_train, y_train, X_val=None):

    y_mean = np.mean(y_train)
    y_train_centered = y_train - y_mean
    
    # Calculate mean and std of X_train
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0, ddof=1)
    
    # Standardize X_train
    X_train_scaled = (X_train - X_mean) / X_std
    
    if X_val is not None:
        X_val_scaled = (X_val - X_mean) / X_std
        return X_train_scaled, y_train_centered, X_val_scaled, y_mean, X_mean, X_std
    
    return X_train_scaled, y_train_centered, y_mean, X_mean, X_std

# ==========================================
# PART A: Closed-Form Ridge
# ==========================================
def ridge_closed_form(X_scaled, y_centered, lam):

    n, p = X_scaled.shape
    I = np.eye(p)
    
    matrix_inv = np.linalg.inv((X_scaled.T @ X_scaled / n) + (lam * I))
    beta = matrix_inv @ (X_scaled.T @ y_centered / n)
    
    return beta

# ==========================================
# PART B: Gradient Descent Ridge
# ==========================================
def ridge_gradient_descent(X_scaled, y_centered, lam, eta=0.1, max_iter=10000, tol=1e-6):

    n, p = X_scaled.shape
    beta = np.zeros(p) # Initialize betas to 0
    
    for _ in range(max_iter):
        error = y_centered - (X_scaled @ beta)
        grad = -(2 / n) * (X_scaled.T @ error) + (2 * lam * beta)
        
        beta_new = beta - (eta * grad)
        
        # Checking for convergence
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
            
        beta = beta_new
        
    return beta

# ==========================================
# PART C: Cross-Validation (From Scratch)
# ==========================================
def ridge_cv(X, y, lambda_grid, k=5):

    n = X.shape[0]
    
    # Shuffling indices for unbiased folds (set seed for reproducibility)
    np.random.seed(42)
    indices = np.arange(n)
    np.random.shuffle(indices)
    
    # Splitting into k folds
    folds = np.array_split(indices, k)
    
    mse_lambda = np.zeros(len(lambda_grid))
    
    for i, lam in enumerate(lambda_grid):
        fold_mses = []
        
        for fold_idx in range(k):
            # Defining validation and training sets for this fold
            val_indices = folds[fold_idx]
            train_indices = np.concatenate([folds[j] for j in range(k) if j != fold_idx])
            
            X_train, y_train = X[train_indices], y[train_indices]
            X_val, y_val = X[val_indices], y[val_indices]
            
            # Standardizing using ONLY fold's training data
            X_train_s, y_train_c, X_val_s, y_mean, _, _ = standardize(X_train, y_train, X_val)
            
            # Fitting model
            beta = ridge_closed_form(X_train_s, y_train_c, lam)
            
            # Predicting on validation data (add y_mean back for the intercept)
            y_pred = (X_val_s @ beta) + y_mean
            
            # Calculating MSE
            mse = np.mean((y_val - y_pred)**2)
            fold_mses.append(mse)
            
        # Averaging MSE across all 5 folds for this specific lambda
        mse_lambda[i] = np.mean(fold_mses)
        
    # Finding the lambda with the lowest validation MSE
    best_idx = np.argmin(mse_lambda)
    best_lam = lambda_grid[best_idx]
    
    return best_lam, mse_lambda

# ==========================================
# TASK 3 FUNCTION: Bootstrap Lasso Stability
# ==========================================
def bootstrap_lasso_stability(X_train, y_train, B=200, random_seed=42):

    n, p = X_train.shape
    
    selection_counts = np.zeros(p)
    
    # Set seed for reproducibility across the B iterations
    np.random.seed(random_seed)
    
    for b in range(B):
        # Draw a bootstrap sample WITH replacement
        boot_indices = np.random.choice(n, size=n, replace=True)
        
        X_boot = X_train[boot_indices]
        y_boot = y_train[boot_indices]
        
        # Standardize predictors using the bootstrap sample's mean and SD
        X_boot_scaled, y_boot_centered, _, _, _ = standardize(X_boot, y_boot)
        
        # Fit Lasso with CV (suppress warnings if it fails to converge on a weird sample)
        lasso_boot = LassoCV(cv=5, max_iter=2000, n_jobs=-1) 
        lasso_boot.fit(X_boot_scaled, y_boot_centered)
        
        # Record nonzero coefficients (|coef| > 1e-8)
        nonzero_mask = np.abs(lasso_boot.coef_) > 1e-8
        
        # Add the boolean mask
        selection_counts += nonzero_mask
        
        # Optional: Print progress every 50 iterations so you know it hasn't frozen
        if (b + 1) % 50 == 0:
            print(f"Bootstrap progress: {b + 1} / {B} loops completed...")
            
    # Calculate selection frequencies (count / B)
    selection_frequencies = selection_counts / B
    
    return selection_frequencies

# ==========================================
# MAIN EXECUTION BLOCK (Runs sequentially)
# ==========================================

# ------------------------------------------
# TASK 1: RIDGE
# ------------------------------------------
best_lambda, cv_mse_array = ridge_cv(X_train_full, y_train_full, lambda_grid, k=5)

X_train_s, y_train_c, X_test_s, y_mean_full, X_mean_full, X_std_full = standardize(
    X_train_full, y_train_full, X_test
)

final_beta_closed = ridge_closed_form(X_train_s, y_train_c, best_lambda)
final_beta_gd = ridge_gradient_descent(X_train_s, y_train_c, best_lambda, eta=0.1)

y_pred_test_closed = (X_test_s @ final_beta_closed) + y_mean_full
y_pred_test_gd = (X_test_s @ final_beta_gd) + y_mean_full

rmse_diff = np.sqrt(np.mean((y_pred_test_closed - y_pred_test_gd)**2))
print(f"Ridge Best Lambda: {best_lambda:.4f}")
print(f"Ridge GD vs Closed-Form RMSE: {rmse_diff:.6f}")

# ------------------------------------------
# TASK 2: LASSO
# ------------------------------------------

# Fit Lasso with Cross-Validation
lasso_model = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso_model.fit(X_train_s, y_train_c)

# Select the tuning parameter (alpha in sklearn is equivalent to lambda)
best_alpha = lasso_model.alpha_

# Extract Coefficients and Analyze Sparsity
lasso_coefs = lasso_model.coef_
nonzero_indices = np.where(np.abs(lasso_coefs) > 1e-8)[0]
n_nonzero = len(nonzero_indices)

selected_vars = [f"x{i+1}" for i in nonzero_indices]

#Predict on Test Data
y_pred_test_lasso = (X_test_s @ lasso_coefs) + y_mean_full

print(f"Lasso Best Alpha (Tuning Parameter): {best_alpha:.4f}")
print(f"Lasso Total Nonzero Coefficients: {n_nonzero} out of 40")
print(f"Lasso Selected Variables: {selected_vars}")


# ------------------------------------------
# TASK 3: BOOTSTRAP STABILITY
# ------------------------------------------
selection_freqs = bootstrap_lasso_stability(X_train_full, y_train_full, B=200)

print("\nCompiling JSON payload...")
results = {
    "ridge_pred_test": y_pred_test_closed.tolist(),
    "ridge_gd_pred_test": y_pred_test_gd.tolist(),
    "ridge_lambda": float(best_lambda),
    "ridge_cv_mse": cv_mse_array.tolist(),
    "ridge_beta": final_beta_closed.tolist(),
    "ridge_gd_beta": final_beta_gd.tolist(),
    "lasso_pred_test": y_pred_test_lasso.tolist(),
    "lasso_alpha": float(best_alpha),
    "lasso_n_nonzero": int(n_nonzero),
    "lasso_selected_vars": selected_vars,
    "lasso_beta": lasso_coefs.tolist(),
    "bootstrap_selection_freq": selection_freqs.tolist()
}

output_file = 'hw4_results_s1467.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=4)

print(f"Execution complete! Results safely saved to {output_file}.")