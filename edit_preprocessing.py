import csv
import numpy as np
import h5py
import json
from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd
import numpy as np

with open('edit_hyper_parameters.json') as json_file:
    parameters = json.load(json_file)
    minlon = parameters['region']['minlon']
    minlat = parameters['region']['minlat']
    maxlon = parameters['region']['maxlon']
    maxlat = parameters['region']['maxlat']
    json_file.close()

def inrange(lon_max,lon_min,lat_max,lat_min):
    if lon_max<maxlon and lon_min>minlon and lat_max<maxlat and lat_min>minlat:
        return True
    else:
        return False

'''
read coordinate data from the CSV file and store them into ndarray
'''

paths = [str(x) for x in Path('autodl-nas/trajSR/data/').glob('data_trj*.csv')]

with h5py.File("./data/dma_traj_array.hdf5",'w') as f:
    loop = tqdm(paths, leave = True)
    
    x = 0
    m = 0
    for i in loop:
        m += 1
        df = pd.read_csv(i)
        # print(list(df))
        df = df.sort_values(by = ['MMSI','time'])
        data = df[['MMSI','Latitude', 'Longitude']]
        data2 = data.groupby('MMSI')['Latitude'].apply(list)
        data3 = data.groupby('MMSI')['Longitude'].apply(list)
        
        i = 0
        for k in range(len(data2)):
            line2 = data2.iloc[k]
            line3 = data3.iloc[k]
            if len(line2) < 180 : continue # 起码要半个小时
            seq = np.zeros((2, len(line2)))
            for j, item in enumerate(line2):
                seq[0, j]= item
            for j, item in enumerate(line3):
                seq[1, j]= item
            lat_max = np.max(seq[0,])
            lat_min = np.min(seq[0,])
            lon_max = np.max(seq[1,])
            lon_min = np.min(seq[1,])
            
            if seq.size <= 1:
                continue
            
            if inrange(lon_max, lon_min, lat_max, lat_min):
                dset = f.create_dataset('%s' % i + '_' + str(m), data=seq, dtype='f')

                i += 1
                m += 1 
                
                # if i % 100 == 0: print("complish traj:{}".format(i))
                # default: if i % 10000 == 0: print("complish:{}/?".format(i))
            #--------------------------------------
    print('total # trajectories:', m)
        
    f.close()

print("Finished")
