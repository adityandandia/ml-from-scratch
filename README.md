# ml-from-scratch

Implementing core ML algorithms from scratch using NumPy — no sklearn, no shortcuts.  
Built to develop deep intuition before using high-level frameworks. Every implementation starts with the math, not the library.

## Implementations

| Algorithm | Status | Key Concepts |
|---|---|---|
| Linear Regression | ✅ Complete | Gradient descent, MSE, weight updates |
| Logistic Regression | ✅ Complete | Sigmoid, binary cross-entropy, decision boundary |
| Neural Network | ✅ Complete | Forward pass, backprop, chain rule |
| Decision Tree | ✅ Complete | Information gain, Gini impurity, recursive splitting |
| Random Forest | ✅ Complete | Bagging, bootstrap sampling, feature randomness |
| Tabular ML (Churn) | ✅ Complete | Pipelines, feature engineering, class imbalance, boosting, SHAP |

## Philosophy

No black boxes. Every project in this series follows the same rule — derive the math, implement it cleanly, then explain it in plain language. The goal was never to use the tools. The goal was to understand what the tools are doing.

Each implementation includes:
- Mathematical derivation before any code
- Clean, readable implementation
- Written explanation of intuition and method

## Stack

- Python 3.x
- NumPy — all from-scratch implementations
- pandas, scikit-learn, imbalanced-learn, lightgbm, shap — Tabular ML final project only

The first five projects are pure NumPy. Tabular ML uses libraries deliberately — the lesson shifts from *how algorithms work* to *how production ML pipelines are engineered*.