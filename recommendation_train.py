import os
import argparse, shutil
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import h5py
import numpy as np
import math
from utils import create_dataset, traj2cell_test_lr, traj2cell_test, draw_lr, draw, get_rank
from dataset import MyDataset, pred_data, db_data
from tqdm.auto import tqdm
import csv
import json
from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd
import random 
import pytorch_ssim
from torch.autograd import Variable
import torch
import torchvision.transforms as T
from diffusion_model import Diffusion
from diffusion_modules import UNet, sr_images
from torchvision.transforms import ToTensor
from trajsim_model import Encoder

parser = argparse.ArgumentParser(description='Train Trajectory Prediction Models')
parser.add_argument('--num_epoch', default=1000, type=int, help='train epoch number')
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--top", type=int, default=4, help="top n similar trajectory used for prediction")
parser.add_argument("--pretrained", default='checkpoint/pred_bestmodel.pt', type=str, help="path to pretrained model (default: none)")

args = parser.parse_args()
print(args)

def readfile(num, filename):
    querydb = h5py.File(filename, 'r')
    db = []
    dbnum = num
    
    for i in tqdm(range(dbnum)):
        db.append(np.array(querydb.get('db/%s'%i), dtype="float64"))
    db = [item for item in db if item.size > 1]
    return db


def inrange(lon_max,lon_min,lat_max,lat_min):
    with open('edit_hyper_parameters.json') as json_file:
        parameters = json.load(json_file)
        minlon = parameters['region']['minlon']
        minlat = parameters['region']['minlat']
        maxlon = parameters['region']['maxlon']
        maxlat = parameters['region']['maxlat']
    json_file.close()
    if lon_max<maxlon and lon_min>minlon and lat_max<maxlat and lat_min>minlat:
        return True
    else:
        return False
    
def create_pred_dataset(data, db):
    with open('edit_hyper_parameters.json') as json_file:
        parameters = json.load(json_file)
        minlon = parameters['region']['minlon']
        minlat = parameters['region']['minlat']
        maxlon = parameters['region']['maxlon']
        maxlat = parameters['region']['maxlat']
    json_file.close()
    
    crop = []
    traj_len = len(data[0]) 
    rand_start = random.randint(0, traj_len - 90) 
    traj_crop = data[0][rand_start : rand_start + 90, :]

    latmax = traj_crop[:, 0].max().numpy()
    latmin = traj_crop[:, 0].min().numpy()
    lonmax = traj_crop[:, 1].max().numpy()
    lonmin = traj_crop[:, 1].min().numpy()
    lr_latmax = latmax + 0.05
    lr_latmin = latmax - 0.05
    lr_lonmax = lonmax + 0.05
    lr_lonmin = lonmax - 0.05
    hr_latmax = latmax + 0.15
    hr_latmin = latmax - 0.15
    hr_lonmax = lonmax + 0.15
    hr_lonmin = lonmax - 0.15

    if latmax + 0.05 > maxlat:
        lr_latmax = maxlat
    elif latmax + 0.15 > maxlat:
        hr_latmax = maxlat

    if latmin - 0.05 < minlat:
        lr_latmin = minlat
    elif latmin - 0.15 < minlat:
        hr_latmin = minlat

    if lonmax + 0.05 > maxlon:
        lr_lonmax = maxlon
    elif lonmax + 0.15 > maxlon:
        hr_lonmax = maxlon

    if lonmin - 0.05 < minlon:
        lr_lonmin = minlon
    elif lonmin - 0.15 < minlon:
        hr_lonmin = minlon

    lr = data[0][(data[0][:, 0] < lr_latmax) & (data[0][:, 0] > lr_latmin) & (data[0][:, 1] < lr_lonmax) & (data[0][:, 1] > lr_lonmin)]
    hr = data[0][(data[0][:, 0] < hr_latmax) & (data[0][:, 0] > hr_latmin) & (data[0][:, 1] < hr_lonmax) & (data[0][:, 1] > hr_lonmin)] 
    db_crop = [lr]
    for j in range(len(db)):
        db_hr = db[j][(db[j][:, 0] < hr_latmax) & (db[j][:, 0] > hr_latmin) & (db[j][:, 1] < hr_lonmax) & (db[j][:, 1] > hr_lonmin)]
        if db_hr.shape[0] == 0:
            continue
        db_crop.append(db_hr)
    crop.append(db_crop)
    return hr, crop
   

def save_checkpoint(state, is_best):
    filename = "checkpoint/pred2_6_checkpoint_epoch_%d.pt" % (state["epoch"])
    if is_best:
        print("##### saving epoch {} as the best model #####".format(state["epoch"]))
        torch.save(state, filename)
        shutil.copyfile(filename, 'checkpoint/test6_pred2_bestmodel.pt') 

def generate_pred_data(name, out_file_name):
    with open('edit_hyper_parameters.json') as json_file:
        parameters = json.load(json_file)
        minlon = parameters['region']['minlon']
        minlat = parameters['region']['minlat']
        maxlon = parameters['region']['maxlon']
        maxlat = parameters['region']['maxlat']
    json_file.close()
    
    paths = []
    for i in name:
        path = [str(x) for x in Path('data/').glob(i)]
        paths.extend(path)

    with h5py.File(out_file_name,'w') as f:
        loop = tqdm(paths, leave = True)

        x = 0
        m = 0
        for i in loop:
            m += 1
            df = pd.read_csv(i)
            df = df.sort_values(by = ['MMSI','time'])
            data = df[['MMSI','Latitude', 'Longitude']]
            data2 = data.groupby('MMSI')['Latitude'].apply(list)
            data3 = data.groupby('MMSI')['Longitude'].apply(list)

            i = 0
            for k in range(len(data2)):
                line2 = data2.iloc[k]
                line3 = data3.iloc[k]
                if len(line2) < 200 : continue 
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
    return m

def main():
    #pred_num = generate_pred_data(['data_trj_202205*.csv'], './data/pred_croptraj_train.hdf5')
    #db_num = generate_pred_data(['data_trj_202204*.csv'], './data/pred_croptraj_train_db.hdf5')

    #print('###',pred_num, 'trajectories used for predction training. ###') # 1290
    #print('###',db_num, 'trajectories used for predction database. ###') # 1277

    #pred_data(pred_num, "./data/pred_croptraj_train.hdf5", "pred_train.hdf5")
    #db_data(db_num, "./data/pred_croptraj_train_db.hdf5", "pred_train_db.hdf5")
    
    #############################################################################################
    # if you go from raw data, please uncomment the 6 lines above and comment the two lines below
    #############################################################################################
    
    pred_num = 1290
    db_numm = 1277
    ###############
    
    pred = readfile(pred_num, "pred_train.hdf5")
    training_idx = int((pred_num*0.8 // 10)*10)
    pred_train = pred[:training_idx]
    pred_val = pred[training_idx:]
    db = readfile(db_num, "pred_train_db.hdf5")

    train_loader = DataLoader(dataset=pred_train, batch_size=1, num_workers=4, shuffle=True)
    val_loader = DataLoader(dataset=pred_val, batch_size=1, num_workers=4, shuffle=False)

    seed = random.randint(1, 10000)
    print("Random Seed: ", seed)
    torch.manual_seed(seed)
    
    netDiffu = Diffusion(noise_steps=500, beta_start=-6, beta_end=6, img_size=256, device="cuda")
    netP = UNet().cuda()
    criterion_l1 = nn.L1Loss(reduction="sum")
    criterion_l1.cuda()
    criterion_mse = nn.MSELoss(reduction="sum")
    criterion_mse.cuda()
    optimizerP = optim.Adam(netP.parameters(), args.lr)
    schedulerP = optim.lr_scheduler.MultiStepLR(optimizerP, milestones=[200, 300], gamma=0.5)
    
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained)
            pred_start_epoch = checkpoint["epoch"]
            val_best_loss = checkpoint["val_best_loss"]
            netP.load_state_dict(checkpoint["prediction"])
            optimizerP.load_state_dict(checkpoint["optimizerP"])
        else:
            print("=> no model found at '{}'".format(args.pretrained))
    else:
        val_best_loss = float('inf')
        pred_start_epoch = 0
    

    #----------load SR Trjsr model--------
    path = 'checkpoint/simdma_bestmodel_MyG_3.pt'
    print("=> loading checkpoint '{}'".format(path))

    netD = Encoder()

    checkpoint = torch.load(path)
    start_epoch = checkpoint["epoch"]
    best_vec_loss = checkpoint["best_loss"]
    netD.load_state_dict(checkpoint["netD"])
    print("the similarity measurement model of epoch %d" % start_epoch)

    if torch.cuda.is_available():
        netD.to(torch.device("cuda:0"))
    else:
        netD.cpu()
    netD.eval()

    
    ##################################################
    # Start training 
    ##################################################

    for epoch in range(1, args.num_epoch-pred_start_epoch):
        # ------------Training---------------
        netP.train()
        train_bar = tqdm(train_loader)

        for num, data in enumerate(train_bar):
            hr, db_crop = create_pred_dataset(data, db)
            db_crop = db_crop[0]

            #---------------------------------------
            # generate output of Trjsr
            traj_vec = []
            traj_img = []
            hr_cell = traj2cell_test(hr)
            hr_img = ToTensor()(draw(hr_cell, index = epoch+pred_start_epoch, mode = 'hr_pred_train')) 
            with torch.no_grad():
                for k in range(len(db_crop)):
                    test_traj = traj2cell_test(db_crop[k])         
                    test_img = ToTensor()(draw(test_traj, index = epoch+pred_start_epoch, mode = 'pred_train'))  
                    traj_img.append(test_img)
                    input = torch.unsqueeze(test_img, 0).cuda()
                    real_out = netD(input)
                    traj_vec.append(real_out)

                #---------------------------------------
                # calculate the similarity 
                dist = []
                for l in range(1, len(traj_vec)):
                    dist.append(np.sum(np.absolute((traj_vec[0].cpu().numpy() - traj_vec[l].cpu().numpy()))))

                score_rank = np.argsort(dist)
                refer = [traj_img[0]]
                for m in range(args.top):
                    if m+1 > len(score_rank):
                        top_traj = np.zeros_like(traj_img[0])
                    else:
                        top_traj = np.array(traj_img[score_rank[m]+1])
                    refer.append(top_traj) # 
                top_sim = np.stack(refer, axis = 0)
            #---------------------------------------
            # Make prediction  
            t = netDiffu.sample_timesteps(hr_img.shape[0]).cuda() 
            x_t, noise = netDiffu.noise_images(hr_img.cuda(), t) 
            top_sim = torch.from_numpy(top_sim)
            predicted_noise = netP(x = x_t.cuda(), lr = top_sim.cuda(), t = t)
            f_loss = criterion_mse(predicted_noise.squeeze(0), noise)
            
            ##############################
            # (3) Update Prediction block:
            ##############################
            netP.zero_grad()
            f_loss.backward()
            optimizerP.step()

            train_bar.set_description(desc='[%d/%d] Training f_loss: %.4f' % (epoch+pred_start_epoch, args.num_epoch, f_loss.item()))


        # ------------Evaluation---------------
        netP.eval()
        with torch.no_grad():
            valing_results = {'f_loss': 0, 'p_loss': 0}
            val_bar = tqdm(val_loader)
            for num, data in enumerate(val_bar):
                val_hr, val_db_crop = create_pred_dataset(data, db)
                val_db_crop = val_db_crop[0]
                #---------------------------------------
                # generate output of Trjsr
                val_traj_vec = []
                val_traj_img = []
                val_hr_cell = traj2cell_test(val_hr)
                val_hr_img = ToTensor()(draw(val_hr_cell, index = epoch+pred_start_epoch, mode = 'hr_pred_val')) 
                for k in range(len(val_db_crop)):
                    val_test_traj = traj2cell_test(val_db_crop[k])         
                    val_test_img = ToTensor()(draw(val_test_traj, index = epoch+pred_start_epoch, mode = 'pred_val'))   
                    val_traj_img.append(val_test_img) 
                    val_input = torch.unsqueeze(val_test_img, 0).cuda()

                    val_real_out = netD(val_input)
                    val_traj_vec.append(val_real_out)

                #---------------------------------------
                # calculate the similarity 
                val_dist = []
                for l in range(1, len(val_traj_vec)):
                    val_dist.append(np.sum(np.absolute((val_traj_vec[0].cpu().numpy() - val_traj_vec[l].cpu().numpy()))))

                val_score_rank = np.argsort(val_dist)
                val_refer = [val_traj_img[0]]
                for m in range(args.top):
                    if m+1 > len(val_score_rank):
                        val_top_traj = np.zeros_like(val_traj_img[0])
                    else:
                        val_top_traj = np.array(val_traj_img[val_score_rank[m]+1])
                    val_refer.append(val_top_traj) 
                val_top_sim = np.stack(val_refer, axis = 0)
                
                #---------------------------------------
                # Make prediction 
                val_t = netDiffu.sample_timesteps(val_hr_img.shape[0]).cuda()
                val_x_t, val_noise = netDiffu.noise_images(val_hr_img.cuda(), val_t.cuda())
                val_top_sim = torch.from_numpy(val_top_sim)
                val_predicted_noise = netP(x = val_x_t.cuda(), lr = val_top_sim.cuda(), t = val_t.cuda())
                val_f_loss = criterion_mse(val_predicted_noise.squeeze(0), val_noise)
                valing_results['f_loss'] += val_f_loss.item()
                if num == 1:
                    val_sr_img = sr_images(xt = val_x_t.cuda(), lr = val_top_sim.cuda(), model = netP, epoch = epoch+pred_start_epoch, mode = 'pred_val_SR')
                    
                    transform = T.ToPILImage()

                val_bar.set_description(desc="Validating f_loss: %.4f" % (val_f_loss.item())) 
            
        batch_val_loss = valing_results['f_loss']/len(val_loader)
        print("epoch:{} batch mean f_loss:{}".format(epoch+pred_start_epoch, batch_val_loss))
        if batch_val_loss < val_best_loss:
            val_best_loss = batch_val_loss
            is_best = True

        else:
            is_best = False

        save_checkpoint({
            "epoch": epoch+pred_start_epoch,
            "val_best_loss": val_best_loss,
            "prediction": netP.state_dict(),
            "optimizerP": optimizerP.state_dict()
        }, is_best)
                
if __name__ == "__main__":
    main()
            
