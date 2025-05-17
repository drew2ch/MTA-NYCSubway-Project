import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Implementing a simple Neural Network Regression model for the Subway Prediction Data using PyTorch.
# Andrew Chung, hc893, 4/27/2025

# import data
data = pd.read_csv("subwaydata.csv").iloc[:, 1:]
X = data.iloc[:, 3:]
y = data['ridership'].to_numpy()/1000 # 1K scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 893)
y_train, y_test = np.log(y_train), np.log(y_test)

# standard scaling before NN implementation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# convert to torch.tensor
X_train_t = torch.tensor(X_train, dtype = torch.float32)
X_test_t = torch.tensor(X_test, dtype = torch.float32)
y_train_t = torch.tensor(y_train, dtype = torch.float32).view(-1, 1)
y_test_t = torch.tensor(y_test, dtype = torch.float32).view(-1, 1)

class SimpleNN(nn.Module):
  def __init__(self):
    super(SimpleNN, self).__init__()
    self.model = nn.Sequential(
      # 24 -> 12 -> 1 architecture
      nn.Linear(X_train.shape[1], 24),
      nn.ReLU(),
      nn.Linear(24, 12),
      nn.ReLU(),
      nn.Linear(12, 1)
    )

  def forward(self, x):
    return self.model(x)

# initialize model (Adam optimizer)
net = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr = 0.001, weight_decay = 1e-4)

# train neural net
for epoch in range(300):
  net.train()
  optimizer.zero_grad()
  outputs = net(X_train_t)
  loss = criterion(outputs, y_train_t)
  loss.backward()
  optimizer.step()

net.eval()
with torch.no_grad():
  y_pred = net(X_test_t).numpy()

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE: {}".format(mse))
print("R²: {}".format(r2))
