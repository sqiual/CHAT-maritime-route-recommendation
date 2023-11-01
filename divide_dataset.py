import h5py
import numpy as np
import math
#from utils import create_dataset
from tqdm.auto import tqdm
import os
import torch


def main(length, obs_len, re_sample_intvl, perc):
    f = h5py.File("data/dma_traj_array.hdf5", 'r')
    # ++++++++++++++++++++++++++++++++++++++++++
    # ++++++++++++++++++++++++++++++++++++++++++
    
    # amount = len(list(f.keys())) // 10 * 10  # the amount of data you expect to use in the experiment
    # ++++++++++++++++++++++++++++++++++++++++++
    # ++++++++++++++++++++++++++++++++++++++++++
    
    num_traj = int(round(len(list(f.keys()))*perc))
    val_index = int(round(0.8 * num_traj))
    # generate trajectory image
    f_names = list(f.keys())
    #print('# of trajectories:', len(list(f.keys())))
    
    if not os.path.exists("data/processed_data/dma_src_train/"):
        os.makedirs("data/processed_data/dma_src_train/")
    if not os.path.exists("data/processed_data/dma_trg_train/"):
        os.makedirs("data/processed_data/dma_trg_train/")
        
    if not os.path.exists("data/processed_data/dma_src_val/"):
        os.makedirs("data/processed_data/dma_src_val/")
    if not os.path.exists("data/processed_data/dma_trg_val/"):
        os.makedirs("data/processed_data/dma_trg_val/")
        
    if not os.path.exists("data/processed_data/dma_whole/"):
        os.makedirs("data/processed_data/dma_whole/")
    
    for i in tqdm(range(int(round(len(list(f.keys()))))), position = 0, leave = True):
        traj = np.array(f.get('%s'%f_names[i]))
        date = f_names[i][-8:]
        if traj.shape[-1] < length:
            continue
        torch.save(traj, "data/processed_data/dma_whole/{}_{}.data".format(str(i), str(date)))
        
    for i in tqdm(range(num_traj), position = 0, leave = True):
        traj = np.array(f.get('%s'%f_names[i]))
        date = f_names[i][-8:]
        if traj.shape[-1] < length:
            continue
        if i < val_index:
            torch.save(traj[:, :obs_len], "data/processed_data/dma_src_train/{}_{}.data".format(str(i), str(date)))
            torch.save(traj[:, obs_len:], "data/processed_data/dma_trg_train/{}_{}.data".format(str(i), str(date)))
            #torch.save(traj, "data/processed_data/dma_whole/{}_{}.data".format(str(i), str(date)))

        else:
            torch.save(traj[:, :obs_len], "data/processed_data/dma_src_val/{}_{}.data".format(str(i), str(date)))
            torch.save(traj[:, obs_len:], "data/processed_data/dma_trg_val/{}_{}.data".format(str(i), str(date)))
            #torch.save(traj, "data/processed_data/dma_whole/{}_{}.data".format(str(i), str(date)))
        #print("data/processed_data/dma_whole/{}_{}.data".format(str(i), str(date)))
        
    print('Total # of trajectories:', len(list(f.keys())))
    print('# of trajectories selected for training:', num_traj)
    print("Finished processing and segmenting trajectories in lengths of ", length*re_sample_intvl/60, "minutes")
    print(val_index, "trajectories for training")
    print(num_traj - val_index, "trajectories for validation")
    f.close()
    
if __name__ == "__main__":
    main(480, 120, 30, 0.1)