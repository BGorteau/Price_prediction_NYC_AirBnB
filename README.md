# Price prediction of New York City's AirBnB
This project present the price prediction of New York City's Air Bnb using Scikit Learn. The main libraries used in this project are `scikit learn`, `pandas` and `matplotlib`. The datas was found on [Kaggle](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data) and a notebook detailing the project is available at `notebooks/price_prediction_NYC_AirBnB.ipynb`.

### Steps of the project:

#### Import and clean the data

The data is imported from a csv file with `Pandas`. 

The different steps to clean the data are :
- Keeping relevant columns
- Replacing `NA` values by the mean of the column
- Doomizing factor variables with `get_dummies`
- Selecting a range to avoid extreme values
- Scaling the data with `StandardScaler`

### Creation of train and test dataframes
We separate our data into train and test dataframes using the function `train_test_split` from `sklearn`.

### Select a metric to compare the different models
We select the Root mean squared error ($RMSE$) to compare our models.

$RMSE = \sqrt{\frac{1}{n}\Sigma_{i=1}^{n}{\Big(\frac{\hat{y_i} -y_i}{\sigma_i}\Big)^2}}$ with $\hat{y_i}$ being the predicted values and $y_i$ the real values.

### Comparison of different models

Here are the models selected and whose performances will be compared.

- Linear models
  * Linear regression (`LinearRegression`)
  * Ridge regression (`Ridge`)
  * Lasso regression (`Lasso`)
- Random forest (`RandomForestRegressor`)
- Gradient boosting (`GradientBoostingRegressor`)

### Model selection

Once we have the RMSEs of the different models, we can compare them and then choose the best model.

| Model                      | RMSE            |
|----------------------------|-----------------|
| `LinearRegression`         | 117.7685        |
| `Ridge`                    | 117.7685        |
| `Lasso`                    | 117.7682        |
| `RandomForestRegressor `   | 109.7626        |
| `GradientBoostingRegressor`| 109.5014        |

When we look at the model, we can see that the three linear models have a veey similar score. In contrast, the Random Forest and Gradient Boosting models have better scores with a slight preference for the Gradient Boosting model.
