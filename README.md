# Statistical Learning Projects

Welcome to my Statistical Learning repository! This profile serves as a dynamic portfolio showcasing core concepts implementations of statistical learning methodologies I learn through my university Statistical Learning course. 

All projects emphasize clean coding standards, reproducibility, and implementing advanced data frameworks from scratch.

# Naming Conventions:

* **hwX_code_s1467** - This is the main code file where I created the model.
* **hwX_slides_code_s1467** - This is the code I used to generate slides for in-class presentation.
* **hwX_test/train_s1467** - This is the data I used to generate the model.
* **hwX_result_s1467** - These are the generated results for the test data.
* **hwX_summary_s1467** - A text file providing a summary of the process.

## 📂 Core Projects

### 🔹 1. Model Selection & Parametric Inference in Multiple Linear Regression
* **Objective:** Identify the true sparse data-generating model from a noisy set of candidate features and generate precise out-of-sample probabilistic predictions.
* **Methodology:** Implemented a full Ordinary Least Squares (OLS) model across five initial features. Conducted a step-down **Backward Elimination** process to prune non-significant predictors ($x_2$ and $x_4$). Verified classical Gauss-Markov assumptions using systematic residual diagnostics, including Normal Q-Q plots, Residuals vs. Fitted plots, and scale-location checks.
* **Key Math/Theory:** Employed partial $t$-tests to isolate the exact true underlying feature subspace ($x_1$, $x_3$, $x_5$). Evaluated model generalization performance using out-of-sample **Negative Log-Likelihood (NLL)**, explicitly measuring the precision of both the estimated conditional mean ($\mu_i$) and the homoscedastic noise variance ($\hat{\sigma}$) against an oracle baseline.

### 🔹 2. Advanced Classification with Feature Interactions
* **Objective:** Predict a balanced binary outcome using continuous predictors where the true decision boundary is highly non-linear.
* **Methodology:** Conducted multivariate EDA to reveal a geometric XOR/checkerboard pattern. Compared Multiple Logistic Regression (with a manually engineered $x_1 \cdot x_3$ interaction term) against Quadratic Discriminant Analysis (QDA).
* **Key Math/Theory:** Utilized the Principle of Marginality to retain non-significant main effects supporting a highly significant interaction. Transformed confidence intervals from the unbounded log-odds scale through a sigmoid link function to ensure valid probability bounds.

### 🔹 3. Bootstrap Methods
* **Objective:** Find the optimal investment allocation weight ($\alpha$) between two correlated variables to minimize total joint variance (risk).
* **Methodology:** Implemented a non-parametric bootstrap entirely from scratch using uniform random number generation and an inverse transformation of the Empirical Distribution Function (EDF). 
* **Key Math/Theory:** Avoided built-in library sampling to execute strict manual index calculation ($\lfloor n \cdot U \rfloor$). Constructed a 95% Confidence Interval using the Bootstrap Percentile Method over 1,000 iterations to test estimator stability.

### 🔹 4. Ridge from Scratch + Lasso with Libraries + Bootstrap Stability
* **Objective:** Address high-dimensional block multicollinearity ($n=150, p=40$) by fitting regularized linear models to control inflated variance and perform automated feature selection.
* **Methodology:** Implemented Ridge Regression ($L_2$ penalty) entirely from scratch using both closed-form matrix algebra and iterative gradient descent. Contrasted this full-architecture shrinkage model with Cross-Validated Lasso Regression ($L_1$ penalty) to enforce strict model sparsity.
* **Key Math/Theory:** Navigated the bias-variance tradeoff via 5-fold CV to isolate optimal penalty parameters ($\lambda$ and $\alpha$). Engineered a 200-iteration Bootstrap resampling simulation to visually expose Lasso's primary theoretical flaw: profound variable selection instability when confronted with highly correlated feature groups.
---

## 🛠️ Tech Stack & Dependencies
The implementations in this repository leverage standard data science and statistical libraries:
* **Data Manipulation:** `pandas`, `numpy`
* **Modeling & Inference:** `statsmodels`, `scikit-learn`
* **Visualization:** `matplotlib`, `seaborn`
