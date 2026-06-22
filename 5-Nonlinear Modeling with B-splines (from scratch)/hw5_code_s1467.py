import numpy as np
import pandas as pd
import json
from sklearn.linear_model import LogisticRegression

# =========================================================
# 1. B-SPLINE CORE (COX-DE BOOR ALGORITHM)
# =========================================================

def evaluate_bspline(x, degree, i, knots):
    """
    Evaluates the i-th B-spline basis function of a given degree at value x.
    """
    if degree == 0:
        if knots[i] <= x < knots[i+1]:
            return 1.0
        elif knots[i] < knots[i+1] and x == knots[-1] and knots[i+1] == knots[-1]:
            return 1.0
        else:
            return 0.0

    denom1 = knots[i + degree] - knots[i]
    term1 = 0.0
    if denom1 > 0:
        term1 = ((x - knots[i]) / denom1) * evaluate_bspline(x, degree - 1, i, knots)

    denom2 = knots[i + degree + 1] - knots[i + 1]
    term2 = 0.0
    if denom2 > 0:
        term2 = ((knots[i + degree + 1] - x) / denom2) * evaluate_bspline(x, degree - 1, i + 1, knots)

    return term1 + term2

def create_knot_vector(degree, m_internal, t_min=0, t_max=40):
    """
    Creates a clamped knot vector bounded by 0 and 40.
    """
    lower_bound = [t_min] * (degree + 1)
    upper_bound = [t_max] * (degree + 1)
    
    if m_internal > 0:
        internal_knots = list(np.linspace(t_min, t_max, m_internal + 2)[1:-1])
    else:
        internal_knots = []
        
    return lower_bound + internal_knots + upper_bound

def build_design_matrix(T, H, W, degree, knots):
    """
    Builds the design matrix: [B_1(T), B_2(T), ..., B_K(T), H, W]
    NO INTERCEPT included to avoid perfect collinearity.
    """
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

# =========================================================
# 2. OLS SOLVER (FROM SCRATCH)
# =========================================================

def fit_ols(X, y):
    return np.linalg.solve(X.T @ X, X.T @ y)

def predict_ols(X, beta):
    return X @ beta

# =========================================================
# 3. CROSS-VALIDATION FOR INTERNAL KNOTS
# =========================================================

def k_fold_cv_for_knots(train_df, max_m=5, k_folds=5, degree=3):
    """
    Finds the optimal number of internal knots (m) using K-Fold CV.
    """
    np.random.seed(42)
    indices = np.random.permutation(len(train_df))
    fold_sizes = len(train_df) // k_folds
    
    best_m = 0
    best_mse = float('inf')
    
    print(f"--- Running {k_folds}-Fold CV for Knot Selection ---")
    
    for m in range(max_m + 1):
        knots = create_knot_vector(degree, m, t_min=0, t_max=40)
        fold_mses = []
        
        for fold in range(k_folds):
            val_idx = indices[fold * fold_sizes : (fold + 1) * fold_sizes]
            train_idx = np.concatenate([indices[:fold * fold_sizes], indices[(fold + 1) * fold_sizes:]])
            
            train_set = train_df.iloc[train_idx]
            val_set = train_df.iloc[val_idx]
            
            X_train = build_design_matrix(train_set['temperature'], train_set['humidity'], train_set['weekend'], degree, knots)
            y_train = train_set['consumption_kwh'].values
            
            X_val = build_design_matrix(val_set['temperature'], val_set['humidity'], val_set['weekend'], degree, knots)
            y_val = val_set['consumption_kwh'].values
            
            beta = fit_ols(X_train, y_train)
            preds = predict_ols(X_val, beta)
            
            mse = np.mean((y_val - preds)**2)
            fold_mses.append(mse)
            
        avg_mse = np.mean(fold_mses)
        print(f"Internal Knots (m={m}): Avg Val MSE = {avg_mse:.4f}")
        
        if avg_mse < best_mse:
            best_mse = avg_mse
            best_m = m
            
    print(f"--> Optimal number of internal knots selected: m = {best_m}")
    return best_m

# =========================================================
# 4. MAIN EXECUTION PIPELINE
# =========================================================

DEGREE = 3

print("Loading datasets...")
train_df = pd.read_csv('hw5_train_s1467.csv')
test_df = pd.read_csv('hw5_test_s1467.csv')

# --- Phase 1: Model Selection ---
best_m = k_fold_cv_for_knots(train_df, max_m=6, k_folds=5, degree=DEGREE)
final_knots = create_knot_vector(DEGREE, best_m, t_min=0, t_max=40)

# --- Phase 2: Build Final Matrices ---
print("\nBuilding final design matrices...")
X_train_full = build_design_matrix(
    train_df['temperature'], train_df['humidity'], train_df['weekend'], DEGREE, final_knots
)
X_test = build_design_matrix(
    test_df['temperature'], test_df['humidity'], test_df['weekend'], DEGREE, final_knots
)

# --- Phase 3: TASK 1 (Regression) ---
print("\n--- Executing Task 1: Regression ---")
y_train_reg = train_df['consumption_kwh'].values
beta_final = fit_ols(X_train_full, y_train_reg)
pred_consumption_kwh = predict_ols(X_test, beta_final)
print("Task 1 Predictions Generated.")

# --- Phase 4: TASK 2 (Classification) ---
print("\n--- Executing Task 2: Classification ---")
y_train_class = train_df['high_demand_alert'].values

# Using Logistic Regression with NO intercept
log_reg = LogisticRegression(fit_intercept=False, penalty=None, max_iter=1000)
log_reg.fit(X_train_full, y_train_class)

pred_high_demand_prob = log_reg.predict_proba(X_test)[:, 1]
pred_high_demand_class = log_reg.predict(X_test)
print("Task 2 Predictions Generated.")

# --- Phase 5: Export JSON ---
print("\n--- Exporting Final Results ---")
submission_data = {
    "pred_consumption_kwh": pred_consumption_kwh.tolist(),
    "pred_high_demand_prob": pred_high_demand_prob.tolist(),
    "pred_high_demand_class": pred_high_demand_class.tolist()
}

output_filename = "hw5_results_s1467.json"
with open(output_filename, "w") as f:
    json.dump(submission_data, f, indent=4)

print(f"Success! Saved all predictions to {output_filename}")