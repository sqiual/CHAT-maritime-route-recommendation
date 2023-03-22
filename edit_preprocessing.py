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

def main():

    '''
    read coordinate data from the CSV file and store them into ndarray
    '''
    name = ['data_trj_202201*.csv', 'data_trj_202202*.csv', 'data_trj_202203*.csv']
    paths = []
    for i in name:
        path = [str(x) for x in Path('./data/').glob(i)]
        paths.extend(path)
    with h5py.File("./data/dma_traj_array.hdf5",'w') as f:
        loop = tqdm(paths, leave = True)

        x = 0
        m = 0
        for i in loop:
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
                if len(line2) < 200: continue 
                seq = np.zeros((2, len(line2)))
                for j, item in enumerate(line2):
                    seq[0, j]= item
                for j, item in enumerate(line3):
                    seq[1, j]= item


                seq = seq[:,((minlat<=seq[0,]) & (seq[0,:]<maxlat) & (minlon<seq[1,:]) & (seq[1,:]<maxlon))]

                
                if seq.size <= 400:
                    continue
                else:
                    i += 1
                    m += 1
                    dset = f.create_dataset('%s' % i + '_' + str(m), data=seq, dtype='f')

                #--------------------------------------
        print('total # trajectories:', m)

        f.close()

    print("Finished")


if __name__ == "__main__":
    main()