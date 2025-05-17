import numpy as np
import pandas as pd
import re
from functools import reduce
from datetime import datetime

# download_url = "https://data.ny.gov/api/views/5wq4-mkjj/rows.csv?accessType=DOWNLOAD"
# hourly ridership (station-level)
hourly_ridership_url = "5wq4-mkjj"

def export_from(url):
  return pd.read_csv("{}{}{}".format(
    "https://data.ny.gov/api/views/",
    url,
    "/rows.csv?accessType=DOWNLOAD"
  ))

def impute(data, 
          gc_shuttle = 'S 42nd', 
          fk_shuttle = 'S Fkln',
          jz = 'J'):
  df = data.copy()
  df.loc[df['line'] == gc_shuttle, 'line'] = 'SG'
  df.loc[df['line'] == fk_shuttle, 'line'] = 'SF'
  df.loc[df['line'] == jz, 'line'] = 'JZ' # merge J/Z
  return df

def extract_lines(text):
  matches = re.findall(r'\((.*?)\)', text)
  items = np.concatenate([item.split(',') for item in matches])
   # sometimes, station names in parentheses get thrown in the mix
  return ','.join(map(str, items[np.char.str_len(items) <= 2]))

def main():

  print("Loading data files from data.ny.gov...")
  print(datetime.now())
  # read in the data: hourly ridership
  hourly_ridership_data = export_from(hourly_ridership_url).query(
    "transit_mode == \'subway\'"
  ).reset_index().drop(columns = ['index'])
  hourly_ridership_data['transit_timestamp'] = pd.to_datetime(hourly_ridership_data['transit_timestamp'])

  print("Hourly Ridership data successfully loaded.")
  print(datetime.now())

  print("Processing hourly ridership data:")

  # cleaning station data

  # define peak time blocks
  start_time_am = pd.to_datetime('06:30:00').time()
  end_time_am = pd.to_datetime('09:30:00').time()
  start_time_pm = pd.to_datetime('15:30:00').time()
  end_time_pm = pd.to_datetime('20:00:00').time()

  # first, filter by month (Jan-Feb)
  hourly_ridership_data = hourly_ridership_data[
    hourly_ridership_data['transit_timestamp'].dt.month < 3
  ]
  # filter hourly ridership data by peak status
  hourly_ridership_data = hourly_ridership_data[
    hourly_ridership_data['transit_timestamp'].dt.time.between(start_time_am, end_time_am) |
    hourly_ridership_data['transit_timestamp'].dt.time.between(start_time_pm, end_time_pm)
  ].sort_values(by = 'transit_timestamp').reset_index().drop(columns = ['index'])\
                                      .groupby(['transit_timestamp', 'station_complex'], as_index = False)\
                                      .agg({
                                        'borough': lambda x: x.mode()[0], # stations do not span different boroughs
                                        'ridership': 'sum'
                                      })
  
  # remove weekends
  hourly_ridership_data = hourly_ridership_data[hourly_ridership_data['transit_timestamp'].dt.weekday < 5] # 5,6 are Sat/Sun
  # remove holidays
  hourly_ridership_data = hourly_ridership_data[~hourly_ridership_data['transit_timestamp'].dt.date.isin(pd.to_datetime([
    '2025-01-01', '2025-01-20', '2025-02-17'
  ]))]

  # group by station ID, aggregate hourly ridership into monthly figures
  hourly_ridership_data['month'] = hourly_ridership_data['transit_timestamp'].dt.month
  monthly_ridership_data = hourly_ridership_data.groupby(['station_complex', 'month'], as_index = False).agg({
    'borough': lambda x: x.mode()[0],
    'ridership': 'sum'
  })
  monthly_ridership_data['lines'] = monthly_ridership_data['station_complex'].apply(extract_lines)
  monthly_ridership_data.to_csv("MTA_Subway_Ridership_Summarized_Apr21.csv", index = False)

  print("Hourly Ridership Dataset successfully written into .csv.")

if __name__ == "__main__":
  main()