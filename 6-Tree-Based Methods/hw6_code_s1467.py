import numpy as np
import pandas as pd
import json
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight

# ==========================================
# 1. Tree Node Class
# ==========================================
class TreeNode:
    def __init__(self, depth):
        self.depth = depth
        self.feature_idx = None  
        self.threshold = None    
        self.left = None         
        self.right = None        
        self.value = None        

# ==========================================
# 2. CART Regressor Class (Task 1)
# ==========================================
class CARTRegressor:
    def __init__(self, max_depth=3, mtry=None):
        self.max_depth = max_depth
        self.mtry = mtry
        self.root = None
        self.feature_importances_ = None 

    def fit(self, X, y):
        self.feature_importances_ = np.zeros(X.shape[1])
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        node = TreeNode(depth=depth)
        
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            node.value = np.mean(y)
            return node
            
        parent_sse = np.sum((y - np.mean(y))**2)
        best_feature, best_threshold, best_sse = self._find_best_split(X, y)
        
        if best_feature is None:
            node.value = np.mean(y)
            return node
            
        decrease = parent_sse - best_sse
        self.feature_importances_[best_feature] += decrease
            
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        node.feature_idx = best_feature
        node.threshold = best_threshold
        
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return node

    def _find_best_split(self, X, y):
        best_feature, best_threshold = None, None
        best_sse = float('inf')
        
        n_samples, n_features = X.shape
        if n_samples <= 1:
            return None, None, float('inf')
            
        if self.mtry is not None:
            feature_indices = np.random.choice(n_features, self.mtry, replace=False)
        else:
            feature_indices = range(n_features)
            
        for feature_idx in feature_indices:
            thresholds = np.unique(X[:, feature_idx])
            
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if not np.any(left_mask) or not np.any(right_mask):
                    continue
                    
                y_left, y_right = y[left_mask], y[right_mask]
                sse_left = np.sum((y_left - np.mean(y_left))**2)
                sse_right = np.sum((y_right - np.mean(y_right))**2)
                total_sse = sse_left + sse_right
                
                if total_sse < best_sse:
                    best_sse = total_sse
                    best_feature = feature_idx
                    best_threshold = threshold
                    
        return best_feature, best_threshold, best_sse

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

# ==========================================
# 3. Custom Random Forest Regressor (Task 2)
# ==========================================
class CustomRandomForestRegressor:
    def __init__(self, n_trees=200, mtry=4, max_depth=None):
        self.n_trees = n_trees
        self.mtry = mtry
        self.max_depth = max_depth
        self.trees = []
        self.feature_importances_ = None
        
    def fit(self, X, y):
        n_samples = X.shape[0]
        self.trees = []
        
        print(f"Building {self.n_trees} trees for the Custom Random Forest...")
        for i in range(self.n_trees):
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[bootstrap_indices]
            y_boot = y[bootstrap_indices]
            
            tree = CARTRegressor(max_depth=self.max_depth, mtry=self.mtry)
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
            
        all_importances = np.array([tree.feature_importances_ for tree in self.trees])
        mean_importances = np.mean(all_importances, axis=0)
        
        if np.sum(mean_importances) > 0:
            self.feature_importances_ = mean_importances / np.sum(mean_importances)
        else:
            self.feature_importances_ = mean_importances
                
    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_preds, axis=0)

# ==========================================
# 4. Cross Validation Function
# ==========================================
def manual_5_fold_cv(X, y, depth_range=range(2, 11)):
    print("Starting 5-Fold CV for CART Regressor...")
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    X_shuf, y_shuf = X[indices], y[indices]
    
    n = len(X)
    fold_sizes = np.full(5, n // 5, dtype=int)
    fold_sizes[:n % 5] += 1 
    
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        X_val, y_val = X_shuf[start:stop], y_shuf[start:stop]
        X_train = np.concatenate([X_shuf[:start], X_shuf[stop:]])
        y_train = np.concatenate([y_shuf[:start], y_shuf[stop:]])
        folds.append((X_train, y_train, X_val, y_val))
        current = stop

    mse_curve = []
    for depth in depth_range:
        fold_mses = []
        for i, (X_train, y_train, X_val, y_val) in enumerate(folds):
            tree = CARTRegressor(max_depth=depth)
            tree.fit(X_train, y_train)
            preds = tree.predict(X_val)
            mse = np.mean((y_val - preds)**2)
            fold_mses.append(mse)
        avg_mse = np.mean(fold_mses)
        mse_curve.append(float(avg_mse)) # Cast to standard float for JSON
    return mse_curve


# ==========================================
# EXECUTION SCRIPT
# ==========================================

# --- Data Loading ---
train_df = pd.read_csv(f'hw6_train_s1467.csv')
test_df = pd.read_csv(f'hw6_test_s1467.csv')

X_train_np = train_df.drop(columns=['sale_price_keur', 'distressed']).values
y_train_reg = train_df['sale_price_keur'].values
y_train_clf = train_df['distressed'].values
X_test_np = test_df.values
feature_names = train_df.drop(columns=['sale_price_keur', 'distressed']).columns

# --- Task 1: Single CART Regressor ---
print("\n=== Task 1: CART Regressor ===")
tree_cv_mse_curve = manual_5_fold_cv(X_train_np, y_train_reg)
best_depth_idx = np.argmin(tree_cv_mse_curve)
best_depth = int(list(range(2, 11))[best_depth_idx]) 

cart_depth3 = CARTRegressor(max_depth=3) 
cart_depth3.fit(X_train_np, y_train_reg)
tree_pred_test = cart_depth3.predict(X_test_np).tolist()

lib_tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=5, criterion='squared_error', random_state=0)
lib_tree.fit(X_train_np, y_train_reg)
tree_lib_pred_test = lib_tree.predict(X_test_np).tolist()


# --- Task 2: Custom Random Forest ---
print("\n=== Task 2: Custom Random Forest ===")
rf_reg_n_trees = 200
rf_reg_mtry = 4

rf = CustomRandomForestRegressor(n_trees=rf_reg_n_trees, mtry=rf_reg_mtry, max_depth=best_depth)
rf.fit(X_train_np, y_train_reg)
rf_reg_pred_test = rf.predict(X_test_np).tolist()
rf_reg_var_importance = {name: float(imp) for name, imp in zip(feature_names, rf.feature_importances_)}


# --- Task 3: Library GBM Regressor ---
print("\n=== Task 3: Library Gradient Boosting Regressor ===")
gbm_reg_base = GradientBoostingRegressor(random_state=42)
param_grid_gbm = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5]
}
grid_search_gbm = GridSearchCV(estimator=gbm_reg_base, param_grid=param_grid_gbm, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_gbm.fit(X_train_np, y_train_reg)

best_gbm_reg = grid_search_gbm.best_estimator_
gbm_reg_hyperparams = grid_search_gbm.best_params_
gbm_reg_pred_test = best_gbm_reg.predict(X_test_np).tolist()
gbm_reg_var_importance = {name: float(imp) for name, imp in zip(feature_names, best_gbm_reg.feature_importances_)}


# --- Task 4: Library Random Forest Classifier ---
print("\n=== Task 4: Library Random Forest Classifier ===")
rf_clf_imbalance_method = "class_weight_balanced"
rf_clf_n_trees = 200
rf_clf_mtry = 4
rf_clf_max_depth = 6 

rf_clf = RandomForestClassifier(n_estimators=rf_clf_n_trees, max_features=rf_clf_mtry, max_depth=rf_clf_max_depth, class_weight='balanced', random_state=42)
rf_clf.fit(X_train_np, y_train_clf)
rf_clf_pred_prob = rf_clf.predict_proba(X_test_np)[:, 1].tolist() # Only probabilities for class 1
rf_clf_var_importance = {name: float(imp) for name, imp in zip(feature_names, rf_clf.feature_importances_)}


# --- Task 5: Library GBM Classifier ---
print("\n=== Task 5: Library GBM Classifier ===")
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_clf)

gbm_clf_hyperparams = {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 3}
gbm_clf = GradientBoostingClassifier(n_estimators=gbm_clf_hyperparams['n_estimators'], learning_rate=gbm_clf_hyperparams['learning_rate'], max_depth=gbm_clf_hyperparams['max_depth'], random_state=42)
gbm_clf.fit(X_train_np, y_train_clf, sample_weight=sample_weights)

gbm_clf_pred_prob = gbm_clf.predict_proba(X_test_np)[:, 1].tolist()
gbm_clf_var_importance = {name: float(imp) for name, imp in zip(feature_names, gbm_clf.feature_importances_)}


# ==========================================
# 6. JSON Compilation and Export
# ==========================================
print("\n=== Generating JSON Output ===")

final_results = {
    "tree_pred_test": tree_pred_test,
    "tree_max_depth": best_depth,
    "tree_cv_mse_curve": tree_cv_mse_curve,
    "tree_lib_pred_test": tree_lib_pred_test,
    
    "rf_reg_pred_test": rf_reg_pred_test,
    "rf_reg_var_importance": rf_reg_var_importance,
    "rf_reg_n_trees": rf_reg_n_trees,
    "rf_reg_mtry": rf_reg_mtry,
    "rf_reg_max_depth": best_depth,
    
    "gbm_reg_pred_test": gbm_reg_pred_test,
    "gbm_reg_hyperparams": gbm_reg_hyperparams,
    "gbm_reg_var_importance": gbm_reg_var_importance,
    
    "rf_clf_pred_prob": rf_clf_pred_prob,
    "rf_clf_var_importance": rf_clf_var_importance,
    "rf_clf_imbalance_method": rf_clf_imbalance_method,
    "rf_clf_n_trees": rf_clf_n_trees,
    "rf_clf_mtry": rf_clf_mtry,
    "rf_clf_max_depth": rf_clf_max_depth,
    
    "gbm_clf_pred_prob": gbm_clf_pred_prob,
    "gbm_clf_var_importance": gbm_clf_var_importance,
    "gbm_clf_hyperparams": gbm_clf_hyperparams
}

json_filename = f"hw6_results_s1467.json"
with open(json_filename, "w") as f:
    json.dump(final_results, f, indent=2)

print(f"Successfully saved all outputs to {json_filename}")