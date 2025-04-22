import numpy as np
import pandas as pd
import re
from functools import reduce
from datetime import datetime

# download_url = "https://data.ny.gov/api/views/5wq4-mkjj/rows.csv?accessType=DOWNLOAD"
# subway line-level data sets
on_time_url = "ks33-g5ze"
trains_delayed_url = "9zbp-wz3y"
service_delivered_url = "nmu4-7tz9"
late_arrivals_url = "x7nj-r656"
major_incidents_url = "uqnw-2qfk"
customer_journey_url = "s4u6-t435"
wait_assessment_url = "62c4-mvcx"
terminal_ontime_url = "ks33-g5ze"

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

  # subway line performance data
  ## customer journey metrics
  customer_journey_data = export_from(customer_journey_url).query(
    "period == \"peak\" & line not in ['W', 'S Rock']"
  ).reset_index().drop(columns = ['division', 'index', 'period'])
  customer_journey_data = impute(customer_journey_data).groupby(['line', 'month']).agg({
    #'num_passengers': 'sum', # sum over 2 months
    'additional platform time': 'mean', 
    'additional train time': 'mean', 
    'over_five_mins_perc': 'mean'
  }).reset_index().rename(columns = {
    'index': 'line',
    'additional platform time': 'additional_platform_time',
    'additional train time': 'additional_train_time'
  })

  ## wait assessment
  wait_assessment_data = export_from(wait_assessment_url).query(
    "day_type == 1 and period == \"peak\" & line not in ['H', 'W', 'S Rock']"
  ).reset_index().drop(columns = ['division', 'index', 'day_type', 'period'])
  wait_assessment_data = impute(wait_assessment_data, gc_shuttle = 'GS', fk_shuttle = 'FS').groupby(['line', 'month']).agg({
    'wait assessment': 'mean'
  }).reset_index().rename(columns = {'index': 'line', 'wait assessment': 'wait_assessment'})

  ## service delivered
  service_delivered_data = export_from(service_delivered_url).query(
    "day_type == 1 & line not in ['H', 'W', 'S Rock']"
  ).reset_index().drop(columns = ['division', 'index', 'day_type'])
  service_delivered_data = impute(service_delivered_data, gc_shuttle = 'GS', fk_shuttle = 'FS').groupby(['line', 'month']).agg({
    'service delivered': 'mean'
  }).reset_index().rename(columns = {'index': 'line', 'service delivered': 'service_delivered'})

  ## terminal on-time performance
  terminal_ontime_data = export_from(terminal_ontime_url).query(
    "day_type == 1 & line not in ['W', 'S Rock']"
  ).reset_index().drop(columns = ['division', 'index', 'day_type'])
  terminal_ontime_data = impute(terminal_ontime_data).groupby(['line', 'month']).agg({
    'terminal_on_time_performance': 'mean'
  }).reset_index().rename(columns = {'index': 'line'})

  # 4-5 minute late arrivala
  late_arrivals_data = export_from(late_arrivals_url).query(
    "`Day Type` == 1 & Line not in ['SI', 'W', 'S Rock']"
  ).reset_index().drop(columns = ['Division', 'index', 'Day Type']).rename(columns = {
    'Month': 'month',
    'Line': 'line',
    'Percent Late': 'percent_late'
  })
  late_arrivals_data = impute(
    late_arrivals_data[late_arrivals_data['month']\
      .isin(['2025-01-01', '2025-02-01'])]\
      .reset_index()\
      .drop(columns = ['index'])
  )
  late_arrivals_data.loc[late_arrivals_data['line'] == 'NW', 'line'] = 'N' # NW -> N
  late_arrivals_data = late_arrivals_data.groupby(['line', 'month']).agg({
    'percent_late': 'mean'
  }).reset_index().rename(columns = {'index': 'line'})

  ## trains delayed
  trains_delayed_data = export_from(trains_delayed_url).query(
    "day_type == 1 & line not in ['W', 'S Rock']"
  ).reset_index().drop(columns = ['division', 'index', 'day_type'])
  trains_delayed_data = impute(trains_delayed_data, gc_shuttle = 'GS')

  ## major incidents
  major_incidents_data = export_from(major_incidents_url).query(
    "day_type == 1"
  ).dropna().reset_index().drop(columns = ['index']).dropna()

  print("All data files loaded.")
  print("Merging First 5 datasets...")

  # aggregate line-specific data

  # lines in the NYC subway system
  subway_lines = np.concatenate((
    np.arange(1,8).astype(str), # numbered lines
    np.array([
      "SG","A","B","C","D","E","F","G","JZ","L","M","N","Q","R","SF"
    ]) # lettered lines
  ))

  # initialize dataset, assign lines and divisions
  line_data = pd.DataFrame(columns =  ['line', 'month']).assign(line = subway_lines, division = None)
  datasets = [
    customer_journey_data, 
    wait_assessment_data,
    service_delivered_data, 
    terminal_ontime_data,
    late_arrivals_data
  ]

  for dataset in range(len(datasets)):
    assert 'line' in datasets[dataset].columns, "line does not exist in {}".format(dataset)

  # Join Datasets
  line_data = reduce(lambda left, right: pd.merge(left, right, on = ['line', 'month'], how = 'left'), datasets)
  line_data

  print("Merging major incidents and train delay data...")

  # major delays and trains delayed data
  # major incidents: indicator variables
  # I will group the incidents into 2 types
  ## 1. Infrastructural -- signal malfunction, subway car, track, stations and structural
  ## 2. Personal/civil: Persons on trackbed/police/medical, other
  incidents = major_incidents_data['category'].unique()
  major_incidents_data['class'] = major_incidents_data['category'].map({
    'Signals': 'Infrastructure',
    'Subway Car': 'Infrastructure',
    'Track': 'Infrastructure',
    'Stations and Structure': 'Infrastructure',
    'Persons on Trackbed/Police/Medical': 'Non-Infrastructure',
    'Other': 'Non-Infrastructure'
  })
  incident_data = major_incidents_data.groupby(['line', 'month', 'class']).agg({
    'count': 'sum'
  }).reset_index().pivot_table(
    index = ['line', 'month'], 
    columns = 'class', 
    values = 'count', 
    aggfunc ='sum'
  ).reset_index().rename(columns = {
    'index': 'line',
    'Infrastructure': 'infra_critical',
    'Non-Infrastructure': 'noninfra_critical'
  }).fillna(0)
  incident_data['infra_critical'] = incident_data['infra_critical'].astype('Int64')
  incident_data['noninfra_critical'] = incident_data['noninfra_critical'].astype('Int64')

  # Delays: in similar fashion, except the reports are already categorized.
  # These events have not spurred major incidents but have nonetheless slowed service.
  ## 1. Infrastructural: Crew Availability, Infra/Equipment, Operating Conditions, Planned ROW work
  ## 2. Non-Infrastructural: Police & Medical, External Factors
  delays = trains_delayed_data['reporting_category'].unique()
  trains_delayed_data['class'] = trains_delayed_data['reporting_category'].map({
    'Crew Availability': 'Infrastructure',
    'Infrastructure & Equipment': 'Infrastructure',
    'Operating Conditions': 'Infrastructure',
    'Planned ROW Work': 'Infrastructure',
    'External Factors': 'Non-Infrastructure',
    'Police & Medical': 'Non-Infrastructure'
  })
  delay_data = trains_delayed_data.groupby(['line', 'month', 'class']).agg({
    'delays': 'sum'
  }).reset_index().pivot_table(
    index = ['line', 'month'],
    columns = 'class',
    values = 'delays',
    aggfunc = 'sum'
  ).reset_index().rename(columns = {
    'index': 'line',
    'Infrastructure': 'infra_noncritical',
    'Non-Infrastructure': 'noninfra_noncritical'
  }).fillna(0)
  delay_data['infra_noncritical'] = delay_data['infra_noncritical'].astype('Int64')
  delay_data['noninfra_noncritical'] = delay_data['noninfra_noncritical'].astype('Int64')

  line_data = line_data.merge(
    incident_data, on = ['line', 'month'], how = 'left'
  ).merge(
    delay_data, on = ['line', 'month'], how = 'left'
  ).fillna(0) # note there is no existing data for major incidents in shuttle services.
  
  line_data.to_csv("MTA_Subway_Line_Data_2025_Apr21.csv")

  print("Line Dataset successfully written into .csv.")

if __name__ == "__main__":
  main()