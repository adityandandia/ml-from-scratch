# ml-from-scratch

Implementing core ML algorithms from scratch using NumPy — no sklearn, no shortcuts.  
Built as part of a structured ML engineering roadmap to develop deep intuition before using high-level frameworks.

## Implementations

| Algorithm | Status | Key Concepts |
|---|---|---|
| Linear Regression | ✅ Complete | Gradient descent, MSE, weight updates |
| Logistic Regression | ✅ Complete | Sigmoid, binary cross-entropy, decision boundary |
| Neural Network | ✅ Complete | Forward pass, backprop, chain rule |
| Decision Tree | ✅ Complete | Information gain, Gini impurity, recursive splitting |
| Random Forest | ✅ Complete | Bagging, bootstrap sampling, feature randomness |
| Tabular ML (Churn) | 🔄 In Progress | Pipelines, feature engineering, class imbalance, boosting |

> This repo concludes with Tabular ML. Deep learning implementations (CNN and beyond) live in a separate PyTorch-based project series.

## Philosophy

Each implementation includes:
- Derivation of the math before any code
- Clean NumPy implementation
- Markdown explanation of intuition and method

## Stack

- Python 3.x
- NumPy only (no ML libraries)