# Principal Component Regression (PCR) Implementation for NYC Subway Ridership
# Andrew Chung, hc893

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def main():
  data = pd.read_csv("subwaydata.csv").iloc[:, 1:]  # remove unlabelled index column
  X = data.iloc[:, 3:]
  y = np.log(data['ridership']/1000) # log-transformed
  # ensure that X and y are equal in size
  assert X.shape[0] == y.size, "Non-conformable X and y inputs"

  # implement PCA regression
  # scaling features
  scaler = StandardScaler()
  scaler.fit_transform(X)

  # fit PCA, regression
  pca = PCA(n_components = 2) # identified during EDA
  X_pca = pca.fit_transform(X)
  pc_cols = [f"PC{i+1}" for i in range(2)]
  df_pca = pd.DataFrame(X_pca, columns=pc_cols)
  ols = sm.OLS(y, sm.add_constant(df_pca)).fit()
  print(ols.summary())

  # EVR plot
  expl_var = np.cumsum(pca.explained_variance_ratio_)
  plt.plot(np.arange(1, len(expl_var)+1), expl_var, marker='o')
  plt.xlabel('Number of PCs')
  plt.ylabel('Cumulative explained variance')
  plt.axhline(0.90, color='gray', linestyle='--')  # e.g. 90% cutoff
  plt.show()

  # regression output
  import seaborn as sns
  # Extract PC-space coefficients (skip intercept)
  beta_pcs = ols.params[1:]  # [β1, β2]
  u = beta_pcs / np.linalg.norm(beta_pcs)

  # Project scores onto this direction to get line range
  t = X_pca.dot(u)
  t_line = np.linspace(t.min(), t.max(), 100)
  line_pts = np.outer(t_line, u)

  # Plot 3D scatter of PC scores + regression direction
  fig = plt.figure()
  ax = fig.add_subplot(111)
  sns.scatterplot(x = X_pca[:, 0], y = X_pca[:, 1])#, X_pca[:, 2])
  ax.plot(line_pts[:, 0], line_pts[:, 1])#, line_pts[:, 2])
  ax.set_xlabel('PC1')
  ax.set_ylabel('PC2')
  #ax.set_zlabel('PC3')
  ax.set_title('PCR')
  plt.show()

if __name__ == "__main__":
  main()