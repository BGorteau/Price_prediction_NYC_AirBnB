"""
train_lasso_reg.py
-------------
Train our data with lasso regression

Author  : Baptiste Gorteau
Date    : May 2025
Project : PRICE_PREDICTION_NYC_AIRBNB
File    : File that train data on lasso regression
"""

#===== IMPORTS =====
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

#===== IMPORT TRAIN AND TEST DATA =====
data = np.load("data/data_train_test.npz")
X_train = data["X_train"]
y_train = data["y_train"]
X_test = data["X_test"]
y_test = data["y_test"]

#===== MODELS PARAMETERS FOR GRID SEARCH =====
alphas = np.array([1,0.4,0.1,0.01,0.001,0.0001,0])
fit_intercept = [True, False]
copy_X = [True, False]
max_iter = [500, 1000, 2000, 3000]
parameters = {"alpha":alphas, "fit_intercept": fit_intercept, "copy_X" : copy_X, "max_iter" : max_iter}

#===== APPLY GRID SEARCH =====
grid_search = GridSearchCV(estimator=Lasso(), 
                           param_grid=parameters,
                           scoring='neg_mean_squared_error',
                           cv=5,
                           n_jobs=-1)

grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_  
best_score = grid_search.best_score_ 

#===== MAKE A MODEL WITH THE BEST PARAMETERS =====
best_model = Lasso(alpha=best_parameters["alpha"],
                    copy_X=best_parameters["copy_X"],
                    fit_intercept=best_parameters["fit_intercept"], 
                    max_iter=best_parameters["max_iter"])

best_model.fit(X_train, y_train)
prediction = best_model.predict(X_test)

#===== COMPUTE THE RMSE =====
rmse = np.sqrt(mean_squared_error(y_test, prediction))

#===== SAVE THE MODEL =====
joblib.dump(best_model, "models/lasso_reg.joblib")

#===== SAVE THE MODEL'S RMSE =====
filename = "results/rmse_performances.json"
if os.path.exists(filename) and os.path.getsize(filename) > 0:
    with open(filename, "r") as f:
        performances_data = json.load(f)
else:
    performances_data = {}

performances_data["Lasso"] = rmse

with open(filename, "w") as f:
    json.dump(performances_data, f, indent=4)
