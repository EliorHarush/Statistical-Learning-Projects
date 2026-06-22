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

### 🔹 5. Nonlinear Modeling with B-splines (from scratch)
* **Objective:** Predict continuous daily household electricity consumption and classify critical high-demand operational days using weather and calendar features exhibiting severe non-linear dynamics.
* **Methodology:** Engineered a flexible, non-linear architecture using cubic B-spline basis functions implemented entirely from scratch via the recursive Cox-de Boor algorithm. Optimized the knot vector ($m=3$) using a custom 5-Fold Cross-Validation pipeline. Evaluated the custom design matrix via Ordinary Least Squares (OLS) built from matrix algebra for regression, and Logistic Regression for classification.
* **Key Math/Theory:** Navigated the mathematical "Partition of Unity" property inherent to B-splines, which necessitated the deliberate exclusion of a global intercept to prevent perfect multicollinearity (matrix singularity). Enforced out-of-sample predictive stability by strictly clamping boundary knots at physical temperature extremes ($0^\circ C$ and $40^\circ C$), and prioritized parsimony by treating secondary features (humidity, weekends) as additive linear effects to avoid high-dimensional overfitting.

### 🔹 6. Tree-Based Methods
* **Objective:** Predict continuous real estate sale prices and classify 'distressed' properties—which exhibit a severe 15% minority class prevalence—using 12 distinct predictors from the Ames Housing dataset.
* **Methodology:** Engineered a CART Regressor (greedy SSE splitting) and a Random Forest Regressor ($B=200$, $m_{try}=4$) entirely from scratch. Benchmarked these custom baseline architectures against hyperparameter-tuned Scikit-Learn Gradient Boosting (GBM) models optimized via 5-Fold Cross-Validation.
* **Key Math/Theory:** Contrasted ensemble variance reduction (Bagging) against sequential bias reduction (Boosting). Calibrated classification probabilities to optimize Cross-Entropy (Log-Loss) rather than naive accuracy, mathematically enforcing minority-class sensitivity via class_weight='balanced' and dynamic sample weights. Extracted intrinsic market drivers using Mean Decrease in Impurity (MDI).

### 🔹 7. Linear SVM on a two-word text classifictaion
* **Objective:** Reverse-engineer a binary text classification problem (Fashion vs. Astronomy) by synthesizing exact training data coordinates to force a linear SVM to learn strictly defined, pre-determined decision boundaries.
* **Methodology:** Modeled text sentences as 2D numerical feature vectors via a Bag-of-Words approach. Synthesized three distinct, linearly separable datasets to evaluate specific spatial constraints (vertical, horizontal, and sloped hyperplanes). Validated the geometric margins using `scikit-learn`'s `SVC` with an exceptionally high penalty parameter ($C=10^5$) to simulate a strict hard-margin environment with zero margin violations.
* **Key Math/Theory:** Manipulated the canonical SVM hyperplane equation ($f(x) = w_1 x_1 + w_2 x_2 + b = 0$) by intentionally designing and plotting synthetic support vectors. Ensured the closest coordinate points rested exactly on the $f(x) = \pm 1$ margins to strictly dictate the learned weights ($w_1, w_2$) and bias ($b$), systematically satisfying complex out-of-sample test accuracy inequalities ($Acc_A > Acc_C > Acc_B$).

### 🔹 8. Unsupervised Learning
* **Objective:** Discover and isolate hidden thematic structures within a completely unlabeled corpus of text documents, deterministically estimating the true number of latent topics without prior ground truth.
* **Methodology:** Engineered a mathematical text representation using Term Frequency-Inverse Document Frequency (TF-IDF), enforcing strict document frequency bounds (min_df, max_df) to isolate core semantic signals from ubiquitous vocabulary. Applied Truncated Singular Value Decomposition (SVD / Latent Semantic Analysis) to compress the highly sparse feature matrix down to 50 principal components. Clustered the reduced embeddings using K-Means with high random initializations ($n\_init=20$) to prevent convergence on suboptimal local minima.
* **Key Math/Theory:** Overcame the "Curse of Dimensionality" inherent to bag-of-words models via SVD to capture robust latent concepts. Solved the unsupervised model selection problem (determining $\hat{K}$) by calculating and maximizing the Silhouette Score across a wide hyperparameter search grid ($K \in [2, 15]$). This explicitly optimized the mathematical balance between intra-cluster cohesion (density) and inter-cluster separation (distance), successfully revealing a global maximum at exactly 11 distinct topic distributions.
---

## 🛠️ Tech Stack & Dependencies
The implementations in this repository leverage standard data science and statistical libraries:
* **Data Manipulation:** `pandas`, `numpy`
* **Modeling & Inference:** `statsmodels`, `scikit-learn`
* **Visualization:** `matplotlib`, `seaborn`
