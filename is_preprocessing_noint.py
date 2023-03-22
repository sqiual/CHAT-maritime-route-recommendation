import pandas as pd
import numpy as np
import os
import math
import pickle
from scipy.interpolate import interp1d
from tqdm.auto import tqdm
import random
from pathlib import Path
# from is_metrics import *
from datetime import datetime



max_len = 300
max_time_diff = 300



def is_data_preprocessing(skip):
    # ===========================================
    # ===========================================
    paths = [str(x) for x in Path('data/').glob('aisdk*.csv')]
    # ===========================================
    # ===========================================
    
    loop = tqdm(paths, leave = True)
    for i in loop:
      print('processing file:', i)
      cleaned = np.empty([1,7])
      df = pd.read_csv(i, on_bad_lines='skip')
      df = df.dropna()
      df['MMSI'] = df['MMSI'].astype(int).astype(str)
      print('Starting with',len(df['MMSI'].unique()), 'trajectories')
      df[['date', 'time']] = df['# Timestamp'].str.split(' ', 1, expand=True)
      df[['day', 'month', 'year']] = df['date'].str.split('/', n=-1, expand=True)
      year = df['year'].iloc[0]
      month = df['month'].iloc[0]
      day = df['day'].iloc[0]
      data = df[['time', 'MMSI', 'Latitude', 'Longitude', 'SOG', 'Navigational status', 'Ship type']]
      data = data[data['MMSI'].apply(lambda x: len(str(x))==9)]
      data['time'] = pd.to_timedelta(data['time']).dt.total_seconds()
    
      # data = data.groupby(['MMSI']).filter(lambda x: x['SOG'].mean() >= min_speed)
      data = data.groupby(['MMSI']).filter(lambda x: ((x['Latitude'].std() != 0) and (x['Longitude'].std() != 0)))

      data = data.sort_values(['MMSI', 'time']).reset_index(drop=True)

      data['sep'] = 0
      data['diff'] = data.groupby('MMSI')['time'].diff()

      data = data.groupby('MMSI', as_index=False).apply(lambda x: x.assign(sep = np.where(x['diff'] > max_time_diff, 1, 0).cumsum()))
    
      data['sep_nav'] = data.groupby(['MMSI'])['Navigational status'].rank(method='dense').astype(int)
      data['MMSI'] = data['MMSI'].astype(str)+data['sep'].astype(str)
      data['MMSI'] = data['MMSI'].astype(str)+data['sep_nav'].astype(str)
      # data = data[data.groupby('MMSI')['time'].transform(lambda x: ((x - x.shift())).max()) < max_time_diff]
      data.groupby('MMSI').filter(lambda x: len(x) > 1) 
        
      #data = data.groupby(['MMSI']).filter(lambda x: x['SOG'].mean() >= min_speed)

      print('Ending with',len(data['MMSI'].unique()), 'trajectories')

      if len(data) == 0:
        continue

      filename = 'data/' + 'data_trj_' + year+month+day +'.csv'

      data.to_csv(filename)
            

    return

is_data_preprocessing(skip=10)
