# Price_prediction_NYC_AirBnB
Price prediction of New York City's Air Bnb with Scikit Learn.

The main libraries used in this project are `scikit learn`, `pandas` and `matplotlib`.

The datas where found on [Kaggle](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data).

## Steps of the project :

### Import and clean the data

The datas are imported form a csv file with `Pandas`. 
The different steps to clean the datas are :
- Keep relevant columns
- Replace NA values by the mean of the column
- Doomizing factor variables
- Select a range to avoid extreme values
- Scale the datas

### Creation of train and test dataframes
We separate our data into train and test datasets using the function `train_test_split` from `sklearn`.

### Select a metric to compare the different models
We select the Root mean squared error (RMSE) to compare our models. 
$RMSE = \sqrt{\frac{1}{n}\Sigma_{i=1}^{n}{\Big(\frac{\hat{y_i} -y_i}{\sigma_i}\Big)^2}}$ with $\hat{y_i}$ being the predicted values and $y_i$ the real values.

### Comparison of different models
- Linear models
  * Linear regression
  * Ridge regression
  * Lasso regression
- Random forest
- Gradient boosting
- PCA

### Model selection
When we got all the RMSE from our models we can compare them and select the best one.
