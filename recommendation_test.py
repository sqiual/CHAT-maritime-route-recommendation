from recommendation_train import generate_pred_data,readfile
from dataset import MyDataset, pred_data, db_data
from utils import create_dataset, traj2cell_test_lr, traj2cell_test, draw_lr, draw
import json
import torch
import torch.nn as nn
import random
from tqdm.auto import tqdm
import numpy as np
import torchvision.transforms as T
from diffusion_model import Diffusion
from diffusion_modules import UNet, sr_images_test
from torchvision.transforms import ToTensor
from trajsim_model import Encoder
import pytorch_ssim
import matplotlib.pyplot as plt

def create_pred_dataset(data, db):
    
    with open('edit_hyper_parameters.json') as json_file:
        parameters = json.load(json_file)
        minlon = parameters['region']['minlon']
        minlat = parameters['region']['minlat']
        maxlon = parameters['region']['maxlon']
        maxlat = parameters['region']['maxlat']
    json_file.close()
    
    crop = []
    traj_len = len(data)
    
    rand_start = random.randint(0, traj_len - 90)
    traj_crop = data[rand_start : rand_start + 90, :]

    latmax = traj_crop[:, 0].max()
    latmin = traj_crop[:, 0].min()
    lonmax = traj_crop[:, 1].max()
    lonmin = traj_crop[:, 1].min()
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

    lr = data[(data[:, 0] < lr_latmax) & (data[:, 0] > lr_latmin) & (data[:, 1] < lr_lonmax) & (data[:, 1] > lr_lonmin)]
    hr = data[(data[:, 0] < hr_latmax) & (data[:, 0] > hr_latmin) & (data[:, 1] < hr_lonmax) & (data[:, 1] > hr_lonmin)] 
    db_crop = [lr]
    for j in range(len(db)):
        db_hr = db[j][(db[j][:, 0] < hr_latmax) & (db[j][:, 0] > hr_latmin) & (db[j][:, 1] < hr_lonmax) & (db[j][:, 1] > hr_lonmin)]
        if db_hr.shape[0] == 0:
            continue
        db_crop.append(db_hr)
    crop.append(db_crop)
    return hr, crop



def main():
    #test_num = generate_pred_data(['data_trj_20220630.csv'], './data/pred_croptraj_test.hdf5') # 46
    #db_num = generate_pred_data(['data_trj_2022062*.csv', 'data_trj_2022061*.csv', 'data_trj_2022060*.csv'], './data/pred_croptraj_test_db.hdf5') # 1205
    #prop = 'gen'  
    #pred_data(test_num, './data/pred_croptraj_test.hdf5', "pred_test.hdf5")
    #db_data(db_num, './data/pred_croptraj_test_db.hdf5', "pred_test_db.hdf5")
    
    #############################################################################################
    # if you go from raw data, please uncomment the 6 lines above and comment the two lines below
    #############################################################################################
    
    pred_num = 46
    db_numm = 1205
    ###############
        

    data = readfile(test_num, "pred_test.hdf5")
    db = readfile(db_num, "pred_test_db.hdf5")

    mae = nn.L1Loss()
    mse = nn.MSELoss()
    transform = T.ToPILImage()
    
    
    #----------load SR Trjsr model--------
    path = 'checkpoint/simdma_bestmodel_MyG_3.pt'
    print("=> loading checkpoint '{}'".format(path))

    netD = Encoder().cuda()

    checkpoint = torch.load(path)
    start_epoch = checkpoint["epoch"]
    best_vec_loss = checkpoint["best_loss"]
    netD.load_state_dict(checkpoint["netD"])
    print("the Trajectory Similarity Measurement model of epoch %d" % start_epoch)

    if torch.cuda.is_available():
        netD.to(torch.device("cuda:0"))
    else:
        netD.cpu()
    netD.eval()

    #-----------load Prediction model---------
    path = 'checkpoint/pred_bestmodel.pt'
    print("=> loading checkpoint '{}'".format(path))
    netDiffu = Diffusion(noise_steps=500, beta_start=-6, beta_end=6, img_size=256, device="cuda")
    netP = UNet().cuda()

    checkpoint = torch.load(path)
    start_epoch = checkpoint["epoch"]
    val_best_loss = checkpoint["val_best_loss"]
    netP.load_state_dict(checkpoint["prediction"])
    print("the Prediction model of epoch %d" % start_epoch)

    if torch.cuda.is_available():
        netP.to(torch.device("cuda:0"))
    else:
        netP.cpu()
    netP.eval()
    
    MAE_loss = []
    MSE_loss = []
    RMSE_loss = []
    SSIM = []
    
    len_data = range(len(data))
    for i in tqdm(len_data, leave = True, position = 0):
        with torch.no_grad():
            hr, db_crop = create_pred_dataset(data[i], db)
            db_crop = db_crop[0]
            #---------------------------------------
            # generate output of Trjsr
            traj_vec = []
            traj_img = []
            hr_cell = traj2cell_test(hr)
            hr_img = ToTensor()(draw(hr_cell, index = 333, mode = 'hr_pred_test')) 

            for k in range(len(db_crop)):
                test_traj = traj2cell_test(db_crop[k])         
                test_img = ToTensor()(draw(test_traj, index = 333, mode = 'pred_test'))  
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
            for m in range(2):
                if m+1 > len(score_rank):
                    top_traj = np.zeros_like(traj_img[0])
                else:
                    top_traj = np.array(traj_img[score_rank[m]+1])
                refer.append(top_traj)# 
            top_traj = np.zeros_like(traj_img[0])
            refer.append(top_traj)
            refer.append(top_traj)
            top_sim = np.stack(refer, axis = 0)

            #---------------------------------------
            # Make prediction  
            t = netDiffu.sample_timesteps(hr_img.shape[0]).cuda() 
            x_t, noise = netDiffu.noise_images(hr_img.cuda(), t) 
            top_sim = torch.from_numpy(top_sim).cuda()
            pred = sr_images_test(xt = x_t.cuda(), lr = top_sim, model = netP, epoch = i)
            hr_img = hr_img.unsqueeze(0)
            
            mae_loss = mae(pred.cuda(), hr_img.cuda())
            mse_loss = mse(pred.cuda(), hr_img.cuda())
            rmse_loss = torch.sqrt(mse_loss)
            MAE_loss.append(mae_loss)
            MSE_loss.append(mse_loss)
            RMSE_loss.append(rmse_loss)
            pred = pred.to(torch.device("cpu"))
            hr_img = hr_img.to(torch.device("cpu"))
        pred = pred.squeeze(0)
        hr_img = hr_img.squeeze(0)
        
    mean_mae_loss = sum(MAE_loss)/len(data)
    mean_mse_loss = sum(MSE_loss)/len(data)
    mean_rmse_loss = sum(RMSE_loss)/len(data)
    print('MAE loss:', mean_mae_loss)
    print('MSE loss:', mean_mse_loss)
    print('RMSE loss:', mean_rmse_loss)
    

if __name__ == "__main__":
    main()
