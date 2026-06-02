import numpy as np
import pandas as pd
from scipy import stats

# -- Load Data --

train = pd.read_csv(r"D:\לימודים\תואר שני\סמסטר ב\Statistical Learning\Projects\1.Linear Regression\Final\hw1_train_s1467.csv")
test = pd.read_csv(r"D:\לימודים\תואר שני\סמסטר ב\Statistical Learning\Projects\1.Linear Regression\Final\hw1_test_s1467.csv")

print(f"Training set: {len(train)} observations, {train.shape[1]-1} predictors")
print(f"Test set: {len(test)} observations")

all_predictors = ["x1", "x2", "x3", "x4", "x5"]
y_train = train["y"].values
n_train = len(y_train)

# -- Full Model --

X_full = train[all_predictors].values
X_full_i = np.column_stack([np.ones(n_train), X_full])

beta_full = np.linalg.lstsq(X_full_i, y_train, rcond=None)[0]
resid_full = y_train - X_full_i @ beta_full
sigma_full = np.sqrt(np.sum(resid_full**2) / (n_train - X_full.shape[1] - 1))

# p-values
XtX_inv = np.linalg.inv(X_full_i.T @ X_full_i)
se_full = sigma_full * np.sqrt(np.diag(XtX_inv))
t_full = beta_full / se_full
p_full = 2 * (1 - stats.t.cdf(np.abs(t_full), df=n_train - X_full.shape[1] - 1))

print("\n=== Full model (all predictors) ===")
names_full = ["intercept"] + all_predictors
for i, name in enumerate(names_full):
    sig = "***" if p_full[i] < 0.001 else "**" if p_full[i] < 0.01 else "*" if p_full[i] < 0.05 else ""
    print(f"  {name:>10s}: beta={beta_full[i]:>8.4f}  SE={se_full[i]:.4f}  "
          f"t={t_full[i]:>7.2f}  p={p_full[i]:.4f} {sig}")
print(f"  sigma_hat = {sigma_full:.4f}")

# -- Variable Selection: Drop variables with p > 0.05

ALPHA = 0.05
selected = [pred for i, pred in enumerate(all_predictors) if p_full[i+1] < ALPHA]
dropped = [pred for pred in all_predictors if pred not in selected]

print(f"\n=== Variable selection (alpha={ALPHA}) ===")
print(f"  Selected predictors: {selected}")
print(f"  Dropped predictors:  {dropped}")

# -- Reduced Model --

X_sel_train = train[selected].values
X_sel_test = test[selected].values
n_sel = len(selected)

X_sel_train_i = np.column_stack([np.ones(n_train), X_sel_train])
X_sel_test_i = np.column_stack([np.ones(len(X_sel_test)), X_sel_test])

beta_sel = np.linalg.lstsq(X_sel_train_i, y_train, rcond=None)[0]
resid_sel = y_train - X_sel_train_i @ beta_sel
sigma_sel = np.sqrt(np.sum(resid_sel**2) / (n_train - n_sel - 1))

print(f"\n=== Reduced model ({', '.join(selected)}) ===")
names_sel = ["intercept"] + selected
for i, name in enumerate(names_sel):
    print(f"  {name:>10s}: beta={beta_sel[i]:>8.4f}")
print(f"  sigma_hat = {sigma_sel:.4f}")

# -- Predictions on Set --
predicted_mean = X_sel_test_i @ beta_sel
predicted_sd = np.full(len(X_sel_test), sigma_sel)

# -- Save Submission --

submission = pd.DataFrame({
    "predicted_mean": np.round(predicted_mean, 4),
    "predicted_sd": np.round(predicted_sd, 4),
})
submission.to_csv(r"D:\לימודים\תואר שני\סמסטר ב\Statistical Learning\Projects\1.Linear Regression\Final\hw1_results_s1467.csv", index=False)
print(f"\nSaved {len(submission)} predictions to D:\לימודים\תואר שני\סמסטר ב\Statistical Learning\Projects\1.Linear Regression\Final\hw1_results_s1467.csv")
