from os import listdir
from os.path import join
import h5py
import numpy as np
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
import torch
import time
from dtaidistance import dtw

# this code is adapted from On Accurate Computation of Trajectory Similarity via Single Image Super-Resolution https://github.com/C-Harlin/trjsr

class MyDataset(Dataset):
    def __init__(self, obs, pred, whole):
        super(MyDataset, self).__init__()
        obs_filenames = [join(obs, x) for x in listdir(obs)]
        pred_filenames = [join(pred, x) for x in listdir(pred)]
        db_filenames = [join(whole, x) for x in listdir(whole)]
        self.obs_filenames = obs_filenames
        self.pred_filenames = pred_filenames
        self.db_filenames = db_filenames

    def __getitem__(self, index):
        obs = torch.load(self.obs_filenames[index])
        pred = torch.load(self.pred_filenames[index])
       
        #obs = obs[:2, :]
        #pred = pred[:2, :]
        obs_org = obs
        pred_org = pred
        origin = obs[:, -1]
        origin_lat = obs[0][0]
        origin_lon = obs[1][0]
        date = self.obs_filenames[index][-13:-5]
        db = []
        obs_zeros = np.zeros((obs.shape[0], 1))

        obs = (obs[:, 1:] - obs[:, :-1])
        
        norm_max_lat = obs[0,:].max()
        norm_min_lat = obs[0:,].min()
        norm_max_lon = obs[1,:].max()
        norm_min_lon = obs[1:,].min()
        
        obs = np.concatenate((obs_zeros, obs), axis = -1)
        #obs = (obs - obs.min())/(obs.max() - obs.min() + 0.000001)
        obs[0,:] = (obs[0,:] - norm_min_lat)/(norm_max_lat - norm_min_lat + 0.000001)
        obs[1,:] = (obs[1,:] - norm_min_lon)/(norm_max_lon - norm_min_lon + 0.000001)
        
        tmp_pred = np.concatenate((origin[:, None], pred), axis = -1)
        pred = (tmp_pred[:, 1:] - tmp_pred[:, :-1])
        #pred = (pred - obs.min())/(obs.max() - obs.min() + 0.000001)
        pred[0,:] = (pred[0,:] - norm_min_lat)/(norm_max_lat - norm_min_lat + 0.000001)
        pred[1,:] = (pred[1,:] - norm_min_lon)/(norm_max_lon - norm_min_lon + 0.000001)
        
        for i in range(len(self.db_filenames)):
            t1 = time.strptime(self.obs_filenames[index][-13:-5],'%Y%m%d')
            t2 = time.strptime(self.db_filenames[i][-13:-5],'%Y%m%d')
            duration = t1.tm_yday - t2.tm_yday
            if (abs(duration) <= 60) and (duration != 0):
                tmp = torch.load(self.db_filenames[i])
                #tmp = tmp[:2, :]
                db_zeros = np.zeros((tmp.shape[0], 1))
                if abs(origin_lat - tmp[0][0] <= 0.005) and abs(origin_lon - tmp[1][0] <= 0.005):
                    tmp = (tmp[:, 1:] - tmp[:, :-1])
                    tmp = np.concatenate((db_zeros, tmp), axis = -1)
                    #tmp = (tmp - obs.min())/(obs.max() - obs.min() + 0.000001)
                    tmp[0,:] = (tmp[0,:] - tmp[0,:].min())/(tmp[0,:].max() - tmp[0,:].min() + 0.000001)
                    tmp[1,:] = (tmp[1,:] - tmp[1,:].min())/(tmp[1,:].max() - tmp[1,:].min() + 0.000001)
                    db.append(tmp)
                else: continue
            else:
                continue

        # https://stackoverflow.com/questions/59738160/denormalization-of-output-from-neural-network
        return obs, pred, obs_org, pred_org, db, origin, norm_max_lat, norm_min_lat, norm_max_lon, norm_min_lon

    def __len__(self):
        return len(self.obs_filenames)