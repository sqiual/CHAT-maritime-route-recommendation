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
            data = df[['MMSI','lat', 'lon', 'width', 'length', 'type']]
            data2 = data.groupby('MMSI')['lat'].apply(list)
            data3 = data.groupby('MMSI')['lon'].apply(list)
            data4 = data.groupby('MMSI')['width'].apply(list)
            data5 = data.groupby('MMSI')['length'].apply(list)
            data6 = data.groupby('MMSI')['type'].apply(list)
            for k in range(len(data2)):
                line2 = data2.iloc[k]
                line3 = data3.iloc[k]
                line4 = data4.iloc[k]
                line5 = data5.iloc[k]
                line6 = data6.iloc[k]
                if len(line2) < length: continue 
                seq = np.zeros((5, len(line2)))
                for x, item in enumerate(line2):
                    seq[0, x]= item
                for x, item in enumerate(line3):
                    seq[1, x]= item
                for x, item in enumerate(line4):
                    seq[2, x]= item
                for x, item in enumerate(line5):
                    seq[3, x]= item
                for x, item in enumerate(line6):
                    'Other', 'Cargo', 'Passenger', 'Tanker', 'HSC'
                    if item == 'Other':
                        seq[4, x]= 0
                    elif item == 'Cargo':
                        seq[4, x]= 1
                    elif item == 'Passenger':
                        seq[4, x]= 2 
                    elif item == 'Tanker':
                        seq[4, x]= 3
                    elif item == 'HSC':
                        seq[4, x]= 4
                    
                #'lat', 'lon', 'width', 'length', 'type'
                seq = seq[:,((minlat<=seq[0,]) & (seq[0,:]<maxlat) & (minlon<seq[1,:]) & (seq[1,:]<maxlon))]
                
                #======================
                #第二次筛选因为限定了经纬度
                #======================
                if seq.shape[-1] < length:
                    continue
                else:
                    window = np.lib.stride_tricks.sliding_window_view(seq, (5, length))[:, ::skip]
                    # skip = 120
                    window = window.squeeze(0)
                    for v in range(window.shape[0]):
                        if window[v].size < length * 5:
                            continue
                        #zero = np.zeros([window[v].shape[0], 1])
                        #tmp = window[v][:, :-1] - window[v][:, 1:]
                        #tmp = np.concatenate((zero, tmp), axis = -1)
                        y += 1
                        # 如果filename是分层的 例如“data/xxx” 这种，那么创建出来的hdf5文件里面会创建出来层级，所有东西会放在“data”这个组下面
                        # 可以参考第六点 https://zhuanlan.zhihu.com/p/396631517
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
    name_test = ['dma_int_pred_2019032*.csv', 'dma_int_pred_2019033*.csv']  # ['dma_int_pred_2019032*.csv', 'dma_int_pred_2019033*.csv'] / ['usa_int_pred_2023022*.csv'] 
    name_db = ['dma_int_pred_201901*.csv','dma_int_pred_201902*.csv','dma_int_pred_201903*.csv'] # ['dma_int_pred_201901*.csv','dma_int_pred_201902*.csv','dma_int_pred_201903*.csv'] / ['usa_int_pred_202301*.csv','usa_int_pred_202302*.csv']
    create_data(name = name_trainval, filename = "data/dma_traj_array.hdf5", length = length, skip = skip)
    create_data(name = name_test, filename = "data/dma_traj_array_test.hdf5", length = length, skip = skip)
    create_data(name = name_db, filename = "data/dma_traj_array_db.hdf5", length = length, skip = skip)


if __name__ == "__main__":
    main(480, 480)