import h5py
import numpy as np
import math
from utils import create_dataset
from tqdm.auto import tqdm


def main():
    f = h5py.File("data/dma_traj_array.hdf5", 'r')

    # ++++++++++++++++++++++++++++++++++++++++++
    # ++++++++++++++++++++++++++++++++++++++++++
    amount = len(list(f.keys())) // 10 * 10  # the amount of data you expect to use in the experiment
    # ++++++++++++++++++++++++++++++++++++++++++
    # ++++++++++++++++++++++++++++++++++++++++++

    val_index = int(0.8 * amount)

    # generate trajectory image
    f_names = list(f.keys())
    print('# of trajectories:', amount)
    for i in tqdm(range(amount), position = 0, leave = True):
        traj = np.array(f.get('%s'%f_names[i]))
        # print('traj.size:', traj.size)
        if traj.size <= 1:
            continue
        if i < val_index:
            create_dataset(traj, i, "train")

        else:
            create_dataset(traj, i-val_index, "val")

    f.close()

if __name__ == "__main__":
    main()