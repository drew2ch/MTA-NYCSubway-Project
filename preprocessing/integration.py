import numpy as np
import pandas as pd

def impute_shuttles(data):
  df = data.copy()
  df = df[~df['lines'].isin(['A,S'])] # remove Rockaway shuttle stations
  # since shuttle type is not specified in hourly ridership data, I will
  # manually impute it based on domain knowledge.

  # SG (42nd St Shuttle) connects to Times Square (1,2,3,7,A,C,E,N,Q,R,W) and Grand Central (4,5,6,7)
  # SF (Franklin Av Shuttle) connects to Franklin Av (A,C), Botanic Garden (2,3,4,5), and Prospect Park (B,Q); Park Place (no connections)

  def assign_lines(val):
    parts = [p.strip() for p in val.split(',')]
    sg_stations = np.array(["N,Q,R,W,S,1,2,3,7,A,C,E", "S,4,5,6,7"])
    sf_stations = np.array(["2,3,4,5,S", "C,S", "B,Q,S", "S"])

    if val in sg_stations:
      return ','.join(['SG' if p == 'S' else p for p in parts])
    elif val in sf_stations:
      return ','.join(['SF' if p == 'S' else p for p in parts])
    else:
      return val

  df['lines'] = df['lines'].apply(assign_lines)
  return df

def main():
  
  print("Processing Monthly Ridership data...")
  # fix shuttle designations
  # aggregate by line complexes
  ridership_data = pd.read_csv(
    "C:\\Users\\hychu\\OneDrive\\Desktop\\SP25\\_BTRY_4100\\_FINAL_PROJECT\\MTA_Subway_Ridership_Summarized_Apr21.csv"
  )
  ridership_data = ridership_data[~ridership_data['station_complex'].isin([
    "Beach 105 St (A,S)", "Beach 25 St (A)", "Beach 36 St (A)", "Beach 44 St (A)", "Beach 67 St (A)", 
    "Beach 90 St (A,S)", "Beach 98 St (A,S)", "Broad Channel (A,S)", "Rockaway Park-Beach 116 St (A,S)"
  ])].copy()
  
  ridership_data = impute_shuttles(ridership_data.groupby(
    ['lines', 'month'], as_index = False).agg({
    'ridership': 'mean'
  }))
  
  print("Processing line performance data...")

  # import line performanace data
  line_data = pd.read_csv("C:\\Users\\hychu\\OneDrive\\Desktop\\SP25\\_BTRY_4100\\_FINAL_PROJECT\\MTA_Subway_Line_Data_2025_Apr21.csv").iloc[:, 1:] # remove index column
  line_data['month'] = pd.to_datetime(line_data['month']).dt.month
  metrics = line_data.columns[3:].to_numpy()
  # compute weights by gross ridership
  line_data['line_weight'] = line_data['num_passengers']/line_data['num_passengers'].sum()

  # assign metrics to ridership data, name it "composite_data_unweighted"
  # no weights for relative subway ridership applied
  composite_data_unweighted = ridership_data.assign(**{col: None for col in metrics})

  # assign weighted metrics to ridership data, name it "composite_data_weighted"
  # weights for relative subway ridership computed by ridership numbers
  composite_data_weighted = ridership_data.assign(**{col: None for col in metrics})

  # helper function to resolve confusions with N/W and J/Z lines
  def impute_lines(lines):
    arr = lines.copy().tolist()
    # Case 1: N/W
    if 'W' in arr:
      if 'N' not in arr:
        arr[arr == 'W'] = 'N'
      else:
        arr.remove('W')

    # Case 2: J/Z
    if 'J' in arr:
      if 'Z' in arr:
        arr.remove('J')
        arr.remove('Z')
        arr.append('JZ')
      else:
        arr = ['JZ' if x == 'J' else x for x in arr]

    return np.array(arr)

  print("Computing unweighted and weighted composite datasets...")

  # fill in performance data (unweighted and weighted have identical indices)
  for idx, row in composite_data_unweighted.iterrows():

    lines_arr = impute_lines(np.array(row['lines'].split(',')))
    row['lines'] = ','.join(lines_arr) # fix line formatting
    month = row['month']
    indices = line_data[
      (line_data['month'] == month) & (line_data['line'].isin(lines_arr))
    ].index.to_numpy()

    for metric in metrics:
      subframe = line_data.loc[indices]
      weights_sum = subframe['line_weight'].sum()
      composite_data_unweighted.loc[idx, metric] = subframe[metric].sum()
      composite_data_weighted.loc[idx, metric] = (
        subframe[metric] * (subframe['line_weight'] * len(indices)/weights_sum)
      ).sum()

  # save to .csv
  composite_data_unweighted.to_csv("Unweighted_Data_Apr23.csv")
  composite_data_weighted.to_csv("Weighted_Data_Apr23.csv")

  print("Successfully saved to .csv.")

if __name__ == "__main__":
  main()