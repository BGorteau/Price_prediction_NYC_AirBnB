"""
compare_models.py
-------------
Compare the results of the different models

Author  : Baptiste Gorteau
Date    : May 2025
Project : PRICE_PREDICTION_NYC_AIRBNB
File    : File that compare the results of the different models
"""

#===== IMPORTS =====
import json
import pandas as pd
import matplotlib.pyplot as plt

#===== IMPORT PERFORMANCES =====
with open("results/rmse_performances.json", "r") as f:
    performances_data = json.load(f)

#===== CREATE A PERFORMANCES DATAFRAME =====
df_performances = pd.DataFrame({"model" : performances_data.keys(), "RMSE" : performances_data.values()})
df_performances = df_performances.sort_values(by=['RMSE'])

#===== PLOT THE RESULTS =====
plt.bar(df_performances["model"], df_performances["RMSE"])
plt.ylim(100,120)
plt.xticks(rotation=30, ha='right')
plt.title("Comparison of model's RMSE")
plt.savefig('figures/rmse_comparison.jpg', dpi=200, bbox_inches="tight")
