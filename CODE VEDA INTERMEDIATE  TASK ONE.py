#!/usr/bin/env python
# coding: utf-8

# # NAME :- VISHAL RAMKUMAR RAJBHAR
# ID :- CV/A1/18203 DOMAIN :- Data Science Intern

# Task 1: Predictive Modeling
# (Regression)
# • Description: Build and evaluate a regression model to
# predict a continuous variable (e.g., house prices).

# In[2]:


# Import libraries needed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# In[3]:


# Load sample dataset
# we can also  replace the datasets by our own datasets but the target value should be updated.
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing(as_frame=True)
df = data.frame

# Features and Target
X = df.drop('MedHouseVal', axis=1)  # features
y = df['MedHouseVal']               # target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# In[4]:


# 1. Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# 2. Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# 3. Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)



# In[5]:


# Evaluation Function
def evaluate_model(name, y_true, y_pred):
    print(f"--- {name} ---")
    print("R² Score:", r2_score(y_true, y_pred))
    print("Mean Squared Error:", mean_squared_error(y_true, y_pred))
    print()

# Evaluate all models
evaluate_model("Linear Regression", y_test, y_pred_lr)
evaluate_model("Decision Tree", y_test, y_pred_dt)
evaluate_model("Random Forest", y_test, y_pred_rf)


# In[6]:


# Plotting Predictions
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.5, label="Random Forest", color='green')
plt.scatter(y_test, y_pred_lr, alpha=0.5, label="Linear Regression", color='blue')
plt.scatter(y_test, y_pred_dt, alpha=0.5, label="Decision Tree", color='orange')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.title("Actual vs Predicted Values")
plt.grid(True)
plt.show()

