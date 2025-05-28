# Decision Tree and XGBoost Regressors for NYC Subway Ridership Prediction
# Andrew Chung, hc893

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

def main():
  data = pd.read_csv("subwaydata.csv").iloc[:, 1:]  # remove unlabelled index column
  X = data.iloc[:, 3:]
  y = data['ridership']/1000
  # ensure that X and y are equal in size
  assert X.shape[0] == y.size, "Non-conformable X and y inputs"

  # train and test split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
  y_train, y_test = np.log(y_train), np.log(y_test)

  # Random Forest Regressor
  rf = RandomForestRegressor(
    n_estimators = 500,
    max_depth = None,
    min_samples_split = 2, 
    min_samples_leaf = 1,
    max_features = 'sqrt', # m = 3-4
    random_state = 42,
    n_jobs = -1
  )

  rf.fit(X_train, y_train)
  rf_preds = rf.predict(X_test)
  rf_mse = mean_squared_error(y_test, rf_preds)
  rf_r2 = r2_score(y_test, rf_preds)

  print("Random Forest Test MSE: {}".format(rf_mse))
  print("Random Forest Test R²: {}".format(rf_r2))

  # XGBoost Regressor
  xgb = XGBRegressor(
      n_estimators=100,    # 100 boosting rounds
      learning_rate=0.1,   # step size shrinkage
      max_depth=3,         # depth of each tree
      subsample=0.8,       # row sampling
      colsample_bytree=0.8, # feature sampling
      reg_alpha=0.1,          # L1 regularization
      reg_lambda=1.0,         # L2 regularization
      random_state=42,
      n_jobs=-1
  )

  xgb.fit(X_train, y_train)
  xgb_preds = xgb.predict(X_test)
  xgb_mse = mean_squared_error(y_test, xgb_preds)
  xgb_r2 = r2_score(y_test, xgb_preds)

  print("XGBoost Test MSE: {}".format(xgb_mse))
  print("XGBoost Test R²: {}".format(xgb_r2))

if __name__ == "__main__":
  main()