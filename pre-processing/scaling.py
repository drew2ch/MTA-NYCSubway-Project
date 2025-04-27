import pandas as pd

# possibly scale the final regression data by number of lines.

def main():

  uwd = pd.read_csv("Unweighted_Data_Apr23.csv").iloc[:, 1:]
  wd = pd.read_csv("Weighted_Data_Apr23.csv").iloc[:, 1:]

  assert uwd.iloc[:, :2].equals(wd.iloc[:, :2]), "Unequal row composition"

  vars = uwd.columns[2:]
  linecounts = uwd['lines'].str.split(',').str.len()
  uwd[vars] = uwd[vars].div(linecounts, axis=0)

  vars = wd.columns[2:]
  linecounts = wd['lines'].str.split(',').str.len()
  wd[vars] = wd[vars].div(linecounts, axis=0)

  uwd.to_csv("Unweighted_Data_Apr26.csv", index = False)
  wd.to_csv("Weighted_Data_Apr26.csv", index = False)

if __name__ == "__main__":
  main()