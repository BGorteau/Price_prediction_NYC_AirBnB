"""
clean_data.py
-------------
Clean data to make it usable

Author  : Baptiste Gorteau
Date    : May 2025
Project : PRICE_PREDICTION_NYC_AIRBNB
File    : File that clean the data
"""

#===== IMPORTS =====
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#===== IMPORT DATA =====
data_ab_ny = pd.read_csv("data/AB_NYC_2019.csv")

#################################################
################# CLEAN THE DATA ################
#################################################

#===== DROP USELESS COLUMNS =====
data_ab_ny = data_ab_ny.drop(["id", "name", "host_name", "host_id", 
                              "last_review", "neighbourhood"], axis=1)

#===== REPLACE NA VALUES BY COLUMN'S MEAN FOR 'reviews_per_month' =====
col_mean = data_ab_ny['reviews_per_month'].mean()
data_ab_ny['reviews_per_month'] = data_ab_ny['reviews_per_month'].fillna(col_mean)

#===== TRANSFORM 'neighbourhood_group' AND room_type' INTO DUMMIES VARIABLES =====
data_ab_ny = pd.get_dummies(data_ab_ny, columns=["neighbourhood_group","room_type"])

#===== KEEP PRICES BELOW 2000 USD =====
data_ab_ny = data_ab_ny[(data_ab_ny["price"])<2000].reset_index(drop=True)

#################################################
########### CREATE TRAIN AND TEST DATA ##########
#################################################

#===== GET X DATA =====
X_ab_ny = data_ab_ny.drop("price", axis=1)

#===== SCALE X DATA =====
scaler = StandardScaler()
scaler.fit(X_ab_ny)
X_ab_ny_scaled = scaler.transform(X_ab_ny)

#===== GET TRAIN AND TEST DATAFRAMES =====
X_train, X_test, y_train, y_test = train_test_split(X_ab_ny_scaled, data_ab_ny["price"], 
                                                    test_size=0.33, random_state=42, 
                                                    shuffle=True)

#===== SAVE THE DATA =====
np.savez("data/data_train_test.npz", X_train=X_train, y_train=y_train.to_numpy(), 
         X_test=X_test, y_test=y_test.to_numpy())
