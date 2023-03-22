from os import listdir
from os.path import join
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, ToTensor, ToPILImage,Resize
import torch
from utils import region

# this code is adapted from On Accurate Computation of Trajectory Similarity via Single Image Super-Resolution https://github.com/C-Harlin/trjsr

class MyDataset(Dataset):
    def __init__(self, hr_dataset_dir, lr_dataset_dir,mode):
        super(MyDataset, self).__init__()
        hr_image_filenames = [join(hr_dataset_dir, x) for x in listdir(hr_dataset_dir)]
        lr_image_filenames = [join(lr_dataset_dir, x) for x in listdir(lr_dataset_dir)]
        # ========= 
        # default 
        # ========= 
        
        if mode =='train':
            # the amount of data used for training
            self.hr_image_filenames = hr_image_filenames[0:2488]
            self.lr_image_filenames = lr_image_filenames[0:2488]
        else:
            # the amount of data used for validation
            self.hr_image_filenames = hr_image_filenames[0:622]
            self.lr_image_filenames = lr_image_filenames[0:622]

    def __getitem__(self, index):
        hr_image = torch.load(self.hr_image_filenames[index])
        lr_image = torch.load(self.lr_image_filenames[index])
        return lr_image, hr_image

    def __len__(self):
        return len(self.hr_image_filenames)


def test_data(num):
    f = h5py.File("./data/dma_traj_array.hdf5", 'r')
    querydb = h5py.File("querydb.hdf5", 'w')

    querynum = 30
    
    print("===> create test dataset")
    f_names = list(f.keys())
    for i in tqdm(range(num-querynum)):
        index = len(f.keys()) - num + i
        traj = np.array(f.get('%s' % f_names[index]))
        traj = np.transpose(traj)
        
        if traj.size <= 1:
            continue
        
        if i <querynum:
            querydb.create_dataset("query/%s" % i, data=traj[::2], dtype='f')
            querydb.create_dataset("db/%s" % i, data=traj[1::2], dtype='f')

        else:
            querydb.create_dataset("db/%s" % i, data=traj[1::2], dtype='f')

    querydb.create_dataset("query/num", data=querynum, dtype='int32')
    querydb.create_dataset("db/num", data=num - querynum, dtype='int32')
    querydb.create_dataset("num", data=num, dtype='int32')
    querydb.close()
    f.close()

    print("finished")
    
def pred_data(num, name, out_file_name):
    f = h5py.File(name, 'r')
    print('Pred data file: ', name , "has ", len(f.keys()), "trajectories")
        
        
    querydb = h5py.File(out_file_name, 'w')
    querynum = 0
    
    print("===> create test dataset")
    f_names = list(f.keys())
    for i in tqdm(range(num)):
        index = len(f.keys()) - num + i
        traj = np.array(f.get('%s' % f_names[index]))
        traj = np.transpose(traj)
        
        if traj.size <= 1:
            continue

        querydb.create_dataset("db/%s" % i, data=traj[1::2], dtype='f')
    querydb.create_dataset("db/num", data=num, dtype='int32')
    querydb.create_dataset("num", data=num, dtype='int32')
    querydb.close()
    f.close()

    print("finished")
    

def db_data(num, name, out_file_name):
    f = h5py.File(name, 'r')
    print('Database file: ', name , "has ", len(f.keys()), "trajectories")

    querydb = h5py.File(out_file_name, 'w')
    querynum = 0

    print("===> create test dataset")
    f_names = list(f.keys())
    for i in tqdm(range(num)):
        index = len(f.keys()) - num + i
        traj = np.array(f.get('%s' % f_names[index]))
        traj = np.transpose(traj)
        
        if traj.size <= 1:
            continue

        querydb.create_dataset("db/%s" % i, data=traj[1::2], dtype='f')
    querydb.create_dataset("db/num", data=num, dtype='int32')
    querydb.create_dataset("num", data=num, dtype='int32')
    querydb.close()
    f.close()

    print("finished")
