# Simple Linear Regression Implementation for NYC Subway Ridership Prediction
# Andrew Chung, hc893

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
from matplotlib import pyplot as plt
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import linear_reset
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

def main():
  data = pd.read_csv("subwaydata.csv").iloc[:, 1:]  # remove unlabelled index column
  X = data.iloc[:, 3:]
  y = data['ridership'].to_numpy()/1000 # scale to 1K
  # ensure that X and y are equal in size
  assert X.shape[0] == y.size, "Incompatible X and y dimensions"

  # run a simple linear regression without any transformations
  ols = sm.OLS(y, sm.add_constant(X)).fit()
  print(ols.summary())

  # model F-test
  f_test = linear_reset(ols, power=2, use_f=True)
  print(f_test)

  # Residual scatter plot
  resid = ols.resid
  y_ = ols.fittedvalues
  plt.figure()
  sns.scatterplot(x = y_, y = resid)
  plt.axhline(0, linestyle='--')
  plt.xlabel('Fitted values')
  plt.ylabel('Residuals')
  plt.title('Residuals vs. Fitted')
  plt.show()

  # log-transformed response simple linear regression
  ols_log = sm.OLS(np.log(y), sm.add_constant(X)).fit()
  ols_log.summary()

  # Residual scatter plot for log-transformed response
  resid = ols_log.resid
  y_ = ols_log.fittedvalues
  plt.figure()
  sns.scatterplot(x = y_, y = resid)
  plt.axhline(0, linestyle='--')
  plt.xlabel('Fitted values')
  plt.ylabel('Residuals')
  plt.title('Residuals vs. Fitted (Log Transformed)')
  plt.show()

  # Q-Q plot for log-transformed response
  sm.qqplot(resid, line = '45')
  plt.title('QQ-plot of residuals')
  plt.show()

  # extract model coefficients
  coefs = pd.concat([ols_log.params, ols_log.conf_int()], axis = 1)
  coefs.columns = ['coeff', 'lower', 'upper']

  # (optional) ridge-regularized regression
  ols_ridge = sm.OLS(np.log(y), sm.add_constant(X)).fit_regularized(method = 'elastic_net', alpha = 0.4641588833612782, L1_wt = 0)
  ols_ridge.summary()

if __name__ == "__main__":
  main()