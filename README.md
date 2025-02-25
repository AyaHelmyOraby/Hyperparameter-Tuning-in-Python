# 📌 Hyperparameter Tuning & Grid Search in Machine Learning

## 📖 Introduction
In machine learning, hyperparameters are external configurations set before training, affecting model performance. Unlike model parameters (learned from data), hyperparameters are manually set and optimized for the best results.

### 🔹 Why Tune Hyperparameters?
- Improve model accuracy
- Reduce overfitting or underfitting
- Enhance generalization on unseen data

### 🔹 Common Hyperparameters
- **Number of trees (n_estimators)** in Random Forest
- **Depth of trees (max_depth)**
- **Learning rate** in Gradient Boosting
- **Number of neighbors (k)** in KNN

---

## 🛠️ Applying Hyperparameter Tuning using Grid Search

### ✅ Step 1: Import Required Libraries
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
```

### ✅ Step 2: Load and Prepare the Dataset
```python
# Load dataset (Example: Iris dataset from scikit-learn)
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels

# Split into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### ✅ Step 3: Define the Model and Initial Hyperparameters
```python
# Define model with default parameters
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Initial Model Accuracy: {accuracy:.2f}")
```

### ✅ Step 4: Define the Hyperparameter Grid
```python
# Define hyperparameter grid for tuning
param_grid = {
    'n_estimators': [50, 100, 200],  # Number of trees in the forest
    'max_depth': [3, 5, 10],  # Maximum depth of trees
    'min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4]  # Minimum samples required at leaf node
}
```

### ✅ Step 5: Perform Grid Search
```python
# Grid Search with Cross-Validation
grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best hyperparameters
print("Best Parameters:", grid_search.best_params_)
```

### ✅ Step 6: Train Model with Best Parameters
```python
# Train model with optimal hyperparameters
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_best = best_model.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred_best)
print(f"Optimized Model Accuracy: {final_accuracy:.2f}")
```

---

## 📌 Conclusion
- **Grid Search** systematically searches for the best combination of hyperparameters.
- It improves model performance by finding the optimal parameter values.
- Using **cross-validation** ensures that the model generalizes well to unseen data.

---

## 📚 Additional Notes
- Other hyperparameter tuning methods include **Random Search** and **Bayesian Optimization**.
- Avoid overfitting by tuning a **small number of parameters first** and gradually refining them.

---

## 🚀 Next Steps
Try applying Grid Search on different models such as:
- **Support Vector Machines (SVMs)**
- **Gradient Boosting Machines (GBMs)**
- **Neural Networks (MLPs)**

Happy Learning! 🎯

