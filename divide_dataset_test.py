import h5py
import numpy as np
import math
#from utils import create_dataset
from tqdm.auto import tqdm
import os
import torch


def main(length, obs_len, re_sample_intvl, perc):
    f = h5py.File("data/dma_traj_array_test.hdf5", 'r')
    f_db = h5py.File("data/dma_traj_array_db.hdf5", 'r')
    # generate trajectory image
    f_names = list(f.keys())
    f_db_names = list(f_db.keys())
    
    if not os.path.exists("data/processed_data/dma_src_test/"):
        os.makedirs("data/processed_data/dma_src_test/")
    if not os.path.exists("data/processed_data/dma_trg_test/"):
        os.makedirs("data/processed_data/dma_trg_test/")
    if not os.path.exists("data/processed_data/dma_whole_test/"):
        os.makedirs("data/processed_data/dma_whole_test/")
    
    for i in tqdm(range(int(round(len(list(f_db.keys()))))), position = 0, leave = True):
        traj = np.array(f_db.get('%s'%f_db_names[i]))
        date = f_db_names[i][-8:]
        if traj.size < length*5:
            continue
        torch.save(traj, "data/processed_data/dma_whole_test/{}_{}.data".format(str(i), str(date)))
        
    num_traj = int(round(len(list(f.keys()))*perc))
    for i in tqdm(range(num_traj), position = 0, leave = True):
        traj = np.array(f.get('%s'%f_names[i]))
        date = f_names[i][-8:]
        if traj.size < length*5:
            continue
        torch.save(traj[:, :obs_len], "data/processed_data/dma_src_test/{}_{}.data".format(str(i), str(date)))
        torch.save(traj[:, obs_len:], "data/processed_data/dma_trg_test/{}_{}.data".format(str(i), str(date)))
        #torch.save(traj, "data/processed_data/dma_whole_test/{}_{}.data".format(str(i), str(date)))
        
    print('Total # of trajectories:', len(list(f.keys())))
    f.close()  
    print('Total # of trajectories:', num_traj)
    print("Finished processing and segmenting trajectories in length of ", length*re_sample_intvl/60, "minutes")

if __name__ == "__main__":
    main(480, 120, 30, 1)