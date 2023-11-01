import csv
import numpy as np
import h5py
import json
from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from math import sin, asin, cos, radians, fabs, sqrt

with open('edit_hyper_parameters.json') as json_file:
    parameters = json.load(json_file)
    minlon = parameters['region']['minlon']
    minlat = parameters['region']['minlat']
    maxlon = parameters['region']['maxlon']
    maxlat = parameters['region']['maxlat']
    json_file.close()

def hav(theta):
    s = sin(theta/2)
    return s * s

def get_distance_hav(lat0, lon0, lat1, lon1):
    lat0 = radians(lat0)
    lon0 = radians(lon0)
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    dlng = fabs(lon0 - lon1)
    dlat = fabs(lat0 - lat1)
    h = hav(dlat) + cos(lat0) * cos(lat1) * hav(dlng)
    distance = 2 * 6371.137 * asin(sqrt(h))
    return distance

def inrange(lon_max,lon_min,lat_max,lat_min):
    if lon_max<maxlon and lon_min>minlon and lat_max<maxlat and lat_min>minlat:
        return True
    else:
        return False
    
def create_data(name, filename, length, skip):
    paths = []
    for i in name:
        path = [str(x) for x in Path('data/').glob(i)]
        paths.extend(path)
    with h5py.File(filename,'w') as f:
        loop = tqdm(paths, leave = True)
        
        y = 0
        for i in loop:
            date = i[-12:-4]
            df = pd.read_csv(i)
            #print('sog', df['sog'].unique())
            # 'time', 'MMSI', 'lat', 'lon', 'sog', 'width', 'length', 'status', 'type'
            df = df.sort_values(by = ['MMSI','time'])
            data = df[['MMSI','lat', 'long']]
            data2 = data.groupby('MMSI')['lat'].apply(list)
            data3 = data.groupby('MMSI')['long'].apply(list)
            for k in range(len(data2)):
                line2 = data2.iloc[k]
                line3 = data3.iloc[k]
                if len(line2) < length: continue 
                seq = np.zeros((2, len(line2)))
                for x, item in enumerate(line2):
                    seq[0, x]= item
                for x, item in enumerate(line3):
                    seq[1, x]= item
                    
                #'lat', 'lon', 'width', 'length', 'type'
                seq = seq[:,((minlat<=seq[0,]) & (seq[0,:]<maxlat) & (minlon<seq[1,:]) & (seq[1,:]<maxlon))]

                if seq.shape[-1] < length:
                    continue
                else:
                    window = np.lib.stride_tricks.sliding_window_view(seq, (2, length))[:, ::skip]
                    # skip = 120
                    window = window.squeeze(0)
                    for v in range(window.shape[0]):
                        if window[v].size < length * 2:
                            continue  # [2, 480]
                        left_lat = window[v][0, :-1]
                        right_lat = window[v][0, 1:]
                        left_lon = window[v][1, :-1]
                        right_lon = window[v][1, 1:]
                        step_speed = []
                        for i in range(window[v].shape[-1]-1):
                            speed = get_distance_hav(left_lat[i], left_lon[i], right_lat[i], right_lon[i])/(30/3600)
                            step_speed.append(speed)

                        if any(ele > 185.2 for ele in step_speed):
                            continue
                        dset = f.create_dataset('%s' % str(y) + '_' + str(date), data=window[v], dtype='f')
                #--------------------------------------
        print('total # trajectories:', y)

    f.close()

    print("Finished")


def main(length, skip):

    '''
    Read coordinate data from the CSV file and store them into ndarray
    Ais messages are 30s apart. If we select traj of 1 hr, that means the traj needs to have 120 datapoints 
    '''
    name_trainval = ['dma_int_pred_201901*.csv', 'dma_int_pred_201902*.csv', 'dma_int_pred_2019030*.csv', 'dma_int_pred_2019031*.csv'] #['dma_int_pred_201901*.csv', 'dma_int_pred_201902*.csv', 'dma_int_pred_2019030*.csv', 'dma_int_pred_2019031*.csv'] / ['dma_int_pred_20190101*.csv']
    #name_trainval =['dma_int_pred_20190101*.csv']
    name_test = ['dma_int_pred_2019032*.csv', 'dma_int_pred_2019033*.csv']  # ['dma_int_pred_2019032*.csv', 'dma_int_pred_2019033*.csv'] / ['usa_int_pred_2023022*.csv'] 
    name_db = ['dma_int_pred_201901*.csv','dma_int_pred_201902*.csv','dma_int_pred_201903*.csv']# ['dma_int_pred_201901*.csv','dma_int_pred_201902*.csv','dma_int_pred_201903*.csv'] / ['usa_int_pred_202301*.csv','usa_int_pred_202302*.csv']
    create_data(name = name_trainval, filename = "data/dma_traj_array.hdf5", length = length, skip = skip)
    create_data(name = name_test, filename = "data/dma_traj_array_test.hdf5", length = length, skip = skip)
    create_data(name = name_db, filename = "data/dma_traj_array_db.hdf5", length = length, skip = skip)


if __name__ == "__main__":
    main(480, 480)
