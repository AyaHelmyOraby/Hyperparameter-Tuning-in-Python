# 📌 Hyperparameter Tuning & Grid Search in Machine Learning
this Link using Iris Dataset

https://colab.research.google.com/drive/1gbrdO_9kutkhSKJneuA2SCPlYOkXjlC_

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


---

# Hyperparameter Tuning and Grid Search in Machine Learning

## Introduction
Hyperparameter tuning is a crucial step in machine learning that optimizes the performance of models by selecting the best set of hyperparameters. One of the most effective techniques for this is **Grid Search**, which systematically evaluates different combinations of hyperparameters to find the optimal configuration.

## Steps to Apply Grid Search in a Machine Learning Dataset

### 1. Data Preparation
**Importance:** Ensuring data is clean and properly formatted improves the accuracy and reliability of the model.
```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data.csv")

# Split into features and target
X = df.drop("target", axis=1)
y = df["target"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 2. Model Selection
**Importance:** Choosing an appropriate algorithm impacts the accuracy and efficiency of predictions.
```python
from sklearn.ensemble import RandomForestClassifier

# Initialize model
model = RandomForestClassifier()
```

### 3. Defining Hyperparameters
**Importance:** The choice of hyperparameters significantly influences model performance.
```python
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
```

### 4. Applying Grid Search
**Importance:** Grid Search systematically explores all hyperparameter combinations to find the best configuration.
```python
from sklearn.model_selection import GridSearchCV

# Initialize Grid Search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')

# Fit the model
grid_search.fit(X_train, y_train)
```

### 5. Analyzing the Best Parameters
**Importance:** Selecting the best hyperparameters ensures optimal model performance.
```python
print("Best parameters found: ", grid_search.best_params_)
print("Best accuracy score: ", grid_search.best_score_)
```

### 6. Evaluating the Final Model
**Importance:** Testing on unseen data helps assess the model’s generalization ability.
```python
from sklearn.metrics import accuracy_score

# Predict on test data
y_pred = grid_search.best_estimator_.predict(X_test)

# Evaluate model
print("Test Accuracy: ", accuracy_score(y_test, y_pred))
```


## 📌 Conclusion
Hyperparameter tuning is an essential part of machine learning, improving model accuracy and efficiency. **Grid Search** is a powerful method for systematically optimizing hyperparameters to achieve the best model performance.
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

