
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. Define the Data
X_A = np.array([[2, 0], [2, 2], [0, 0], [0, 2]])
y_A = np.array([1, 1, -1, -1])

X_B = np.array([[0, 2], [2, 2], [0, 0], [2, 0]])
y_B = np.array([1, 1, -1, -1])

X_C = np.array([[2, 1], [1, 3], [1, 1], [0, 3]])
y_C = np.array([1, 1, -1, -1])

X_test = np.array([
    [2, 1], [3, 0], [2, 0], [2, 2],  # +1 (Fashion)
    [0, 5], [0, 0], [0, 1], [0, 0]   # -1 (Astronomy)
])
y_test = np.array([1, 1, 1, 1, -1, -1, -1, -1])

# 2. Train Hard-Margin SVMs (C=1e5 enforces no margin violations)
svm_A = SVC(kernel='linear', C=1e5).fit(X_A, y_A)
svm_B = SVC(kernel='linear', C=1e5).fit(X_B, y_B)
svm_C = SVC(kernel='linear', C=1e5).fit(X_C, y_C)

# Extract Coefficients
def get_coeffs(svm_model):
    w = np.round(svm_model.coef_[0], 2)
    b = np.round(svm_model.intercept_[0], 2)
    return w[0], w[1], b

w1_A, w2_A, b_A = get_coeffs(svm_A)
w1_B, w2_B, b_B = get_coeffs(svm_B)
w1_C, w2_C, b_C = get_coeffs(svm_C)

print("--- SVM Coefficients ---")
print(f"Set A: w1={w1_A}, w2={w2_A}, b={b_A}")
print(f"Set B: w1={w1_B}, w2={w2_B}, b={b_B}")
print(f"Set C: w1={w1_C}, w2={w2_C}, b={b_C}\n")

# 3. Verify Accuracies
acc_A = accuracy_score(y_test, svm_A.predict(X_test))
acc_B = accuracy_score(y_test, svm_B.predict(X_test))
acc_C = accuracy_score(y_test, svm_C.predict(X_test))

print("--- Test Accuracies ---")
print(f"Accuracy A: {acc_A * 100}%")
print(f"Accuracy B: {acc_B * 100}%")
print(f"Accuracy C: {acc_C * 100}%")
print(f"Condition Acc_A > Acc_C > Acc_B met? {acc_A > acc_C > acc_B}")

# 4. EDA Plots for the synthetic sets
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
sets = [(X_A, y_A, svm_A, 'Set A'), (X_B, y_B, svm_B, 'Set B'), (X_C, y_C, svm_C, 'Set C')]

for ax, (X, y, model, title) in zip(axes, sets):
    # Plot points
    ax.scatter(X[y==1][:, 0], X[y==1][:, 1], color='blue', label='+1 (Fashion)', s=100)
    ax.scatter(X[y==-1][:, 0], X[y==-1][:, 1], color='red', label='-1 (Astronomy)', s=100)
    
    # Plot decision boundary
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    
    ax.set_title(title)
    ax.set_xlabel('Word 1 Count (fabric)')
    ax.set_ylabel('Word 2 Count (trend)')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.show()