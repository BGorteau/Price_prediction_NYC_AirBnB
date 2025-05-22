"""
train_gb_reg.py
-------------
Train our data with gradient boosting regressor

Author  : Baptiste Gorteau
Date    : May 2025
Project : PRICE_PREDICTION_NYC_AIRBNB
File    : File that train data on gradient boosting regressor
"""

#===== IMPORTS =====
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

#===== IMPORT TRAIN AND TEST DATA =====
data = np.load("data/data_train_test.npz")
X_train = data["X_train"]
y_train = data["y_train"]
X_test = data["X_test"]
y_test = data["y_test"]

#===== MODELS PARAMETERS FOR GRID SEARCH =====
learning_rate = [0.01, 0.1, 0.5]
min_samples_split = [2,5,10, 20]
n_estimators = [100, 300]

parameters = {"learning_rate" : learning_rate, 
                "min_samples_split" : min_samples_split, 
                "n_estimators" : n_estimators}

#===== APPLY GRID SEARCH =====
grid_search = GridSearchCV(estimator=GradientBoostingRegressor(), 
                           param_grid=parameters,
                           scoring='neg_mean_squared_error',
                           cv=5,
                           n_jobs=-1)

grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_  
best_score = grid_search.best_score_ 

#===== MAKE A MODEL WITH THE BEST PARAMETERS =====
best_model = GradientBoostingRegressor(learning_rate = best_parameters["learning_rate"], 
                                        min_samples_split = best_parameters["min_samples_split"], 
                                        n_estimators = best_parameters["n_estimators"])

best_model.fit(X_train, y_train)

prediction = best_model.predict(X_test)

#===== COMPUTE THE RMSE =====
rmse = np.sqrt(mean_squared_error(y_test, prediction))

#===== SAVE THE MODEL =====
joblib.dump(best_model, "models/gb_reg.joblib")

#===== SAVE THE MODEL'S RMSE =====
filename = "results/rmse_performances.json"
if os.path.exists(filename) and os.path.getsize(filename) > 0:
    with open(filename, "r") as f:
        performances_data = json.load(f)
else:
    performances_data = {}

performances_data["GradientBoostingRegressor"] = rmse

with open(filename, "w") as f:
    json.dump(performances_data, f, indent=4)
