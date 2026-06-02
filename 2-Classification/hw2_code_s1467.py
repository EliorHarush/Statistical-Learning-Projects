
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                             roc_auc_score, roc_curve, precision_score, recall_score, f1_score)
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# -- Load Data --
train_df = pd.read_csv('hw2_train_s1467.csv')
test_df = pd.read_csv('hw2_test_s1467.csv')

# -- Data Manipulation (adding interaction term) --
train_df['x1x3'] = train_df['x1'] * train_df['x3']
test_df['x1x3'] = test_df['x1'] * test_df['x3']

# -- Data Preperation --
features = ['x1', 'x2', 'x3', 'x1x3']
X = train_df[features]
y = train_df['y']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# -- Logistic Regression --
X_train_logit = sm.add_constant(X_train)
X_val_logit = sm.add_constant(X_val)

logit_model = sm.Logit(y_train, X_train_logit).fit()

# -- Validation Predictions --
y_prob_logit = logit_model.predict(X_val_logit)
y_pred_logit = (y_prob_logit > 0.5).astype(int)

# -- QDA --
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train[['x1', 'x2', 'x3']], y_train)

# -- Validation Predictions --
y_prob_qda = qda.predict_proba(X_val[['x1', 'x2', 'x3']])[:, 1]
y_pred_qda = qda.predict(X_val[['x1', 'x2', 'x3']])

# -- Comparing Metrics --
def print_metrics(y_true, y_pred, y_prob, label):
    print(f"\n--- {label} METRICS ---")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"AUC: {roc_auc_score(y_true, y_prob):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

print_metrics(y_val, y_pred_logit, y_prob_logit, "LOGISTIC REGRESSION")
print_metrics(y_val, y_pred_qda, y_prob_qda, "QDA")

# -- Final Prediction using Logistic Regression --
X_test_final = sm.add_constant(test_df[features])

# Compute log-odds: η = X * β
log_odds = np.dot(X_test_final, logit_model.params)

# Compute standard error of log-odds: SE = sqrt(diag(X * Cov * X'))
cov_matrix = logit_model.cov_params()
se_log_odds = np.sqrt(np.sum(np.dot(X_test_final, cov_matrix) * X_test_final, axis=1))

# 95% CI on Log-Odds scale
z_crit = 1.96
log_odds_lower = log_odds - z_crit * se_log_odds
log_odds_upper = log_odds + z_crit * se_log_odds

# Define Sigmoid function for transformation
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Transform back to Probability scale
prob_class1 = sigmoid(log_odds)
ci_lower = sigmoid(log_odds_lower)
ci_upper = sigmoid(log_odds_upper)

# 6. SAVE RESULTS
results_csv = pd.DataFrame({
    'prob_class1': prob_class1,
    'ci_lower': ci_lower,
    'ci_upper': ci_upper
})

print(logit_model.summary())
results_csv.to_csv('hw2_results_s1467.csv', index=False)
print("\nFinal results saved to 'hw2_results_s1467.csv'.")