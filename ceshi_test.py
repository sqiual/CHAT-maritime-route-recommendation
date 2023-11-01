from dataset_test import MyDataset
import json
import torch
import torch.nn as nn
import random
from tqdm.auto import tqdm
import numpy as np
from model import Encoder, TemporalConvNet
import pytorch_ssim
import matplotlib.pyplot as plt
import time
from torch.utils.data import DataLoader
from math import sin, asin, cos, radians, fabs, sqrt
from shapely.geometry import box
import geopandas as gpd
from shapely.geometry import Point
import pickle

#https://blog.csdn.net/chinawangfei/article/details/125458095

with open('edit_hyper_parameters.json') as json_file:
    parameters = json.load(json_file)
    minlon = parameters['region']['minlon']
    minlat = parameters['region']['minlat']
    maxlon = parameters['region']['maxlon']
    maxlat = parameters['region']['maxlat']
    json_file.close()
    

def reconstruct(model_pred_test, pred_test, obs_in, pred_in, origin, norm_max_lat, norm_min_lat, norm_max_lon, norm_min_lon):
    model_pred_test[:,0,:] = model_pred_test[:,0,:] * (norm_max_lat - norm_min_lat + 0.000001) + norm_min_lat
    model_pred_test[:,1,:] = model_pred_test[:,1,:] * (norm_max_lon - norm_min_lon + 0.000001) + norm_min_lon
    pred_test[:,0,:] = pred_test[:,0,:] * (norm_max_lat - norm_min_lat + 0.000001) + norm_min_lat
    pred_test[:,1,:] = pred_test[:,1,:] * (norm_max_lon - norm_min_lon + 0.000001) + norm_min_lon
    for i in range(pred_in):
                if i == 0:
                    model_pred_test[:, :, int(obs_in+i)] += origin.cuda()
                    pred_test[:, :, int(i)] += origin.cuda()
                else:
                    model_pred_test[:, :, int(obs_in+i)] += model_pred_test[:, :, int(obs_in+i-1)]
                    pred_test[:, :, int(i)] += pred_test[:, :, int(i-1)]
    return model_pred_test, pred_test
    
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
    
def main(draw = False):
    # ========== #
    # ========== #

    # ========== #
    # ========== #
    c_in=2
    obs_in = 120
    pred_in = 360
    db_in = 480
    top_num = 9
    noise_steps = 500
    tcn_hidden = [2,2,2,2,2]
    batch_size = 1
    
    test_set = MyDataset('data/processed_data/dma_src_test', 'data/processed_data/dma_trg_test', 'data/processed_data/dma_whole_test')
    
    
    #----------load CHAT model--------
    netE = Encoder(c_in=c_in, obs_in = obs_in, pred_in = pred_in, db_in = db_in, top_num = top_num, tcn_hidden = tcn_hidden).cuda()
    #netDiffu = Diffusion(noise_steps=noise_steps, beta_start=-6, beta_end=6, img_size=256, device="cuda")
    #netP = TemporalConvNet(num_inputs = c_in, num_channels = tcn_hidden).cuda()
    
    #path = "checkpoint/CHAT_star6_antiTCN_1mon_top9_epoch_4.pt"
    path = 'checkpoint/CHAT_star6_BTCN_top9_bestmodel.pt'
    print("=> loading checkpoint '{}'".format(path))
    
    checkpoint = torch.load(path)
    pred_start_epoch = checkpoint["epoch"]
    netE.load_state_dict(checkpoint["Encoder"])
    #optimizerE.load_state_dict(checkpoint["optimizerE"])
    #netP.load_state_dict(checkpoint["Prediction"])
    #optimizerP.load_state_dict(checkpoint["optimizerP"])
    
    print(path[11:20], "model of epoch %d" % pred_start_epoch)

    if torch.cuda.is_available():
        netE.to(torch.device("cuda:0"))
        #netP.to(torch.device("cuda:0"))
    else:
        netE.cpu()
        #netP.cpu()
    
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size)
  
    netE.eval()
    #netP.eval()
    
    inf_time = []
    with torch.no_grad():
        test_results = {'ADE_avg_025h' : 0,'ADE_avg_050h' : 0,'ADE_avg_075h' : 0,'ADE_avg_1h' : 0,'ADE_avg_125h' : 0,'ADE_avg_150h' : 0,'ADE_avg_175h' : 0, 'ADE_avg_2h' : 0, 'ADE_avg_225h' : 0,'ADE_avg_250h' : 0,'ADE_avg_275h' : 0,'ADE_avg_3h' : 0, 'ADE_avg_325h' : 0,'ADE_avg_350h' : 0,'ADE_avg_375h' : 0,'FDE_025h': 0,'FDE_050h': 0,'FDE_075h': 0,'FDE_1h': 0, 'FDE_125h': 0,'FDE_150h': 0,'FDE_175h': 0,'FDE_2h': 0, 'FDE_225h': 0,'FDE_250h': 0,'FDE_275h': 0,'FDE_3h': 0}
        test_bar = tqdm(test_loader)
        for num, data in enumerate(test_bar):
            starttime = time.time()
            obs_test, pred_test, obs_org_test, pred_org_test, db_test, origin, norm_max_lat, norm_min_lat, norm_max_lon, norm_min_lon = data

            whole_test = torch.cat((obs_test, pred_test), dim = -1) 
            #---------------------------------------
            # Begin Encoding
            top_sim_emb_test, top_sim_org_test = netE(obs_test, pred_test, db_test)
            
            #---------------------------------------
            # Make prediction  
            #t_test = netDiffu.sample_timesteps(whole_emb_test.shape[0]).cuda() 
            #x_t_test, noise_test = netDiffu.noise_images(whole_test.cuda(), t_test) 
            
            #model_pred_test = pred_traj(xt = x_t_test.cuda(), cond = top_sim_emb_test.cuda(), model = netP.cuda(), step = noise_steps)
            #model_pred_test = netP(top_sim_org_test)
            model_pred_test = top_sim_org_test   #[1, 2, 480]
            pred_test = pred_test.cuda()         #[1, 2, 360]
            endtime = time.time()
            inf_time.append(endtime - starttime)
            model_pred_test, pred_test = reconstruct(model_pred_test, pred_test, obs_in, pred_in, origin, norm_max_lat.cuda(), norm_min_lat.cuda(), norm_max_lon.cuda(), norm_min_lon.cuda())
            
            #======
            model_pred_test = model_pred_test[:, :, obs_in:]
            model_pred_test_025h = model_pred_test[:, :, :30]
            model_pred_test_050h = model_pred_test[:, :, :60]
            model_pred_test_075h = model_pred_test[:, :, :90]
            model_pred_test_1h = model_pred_test[:, :, :120]
            model_pred_test_125h = model_pred_test[:, :, :30+120]
            model_pred_test_150h = model_pred_test[:, :, :60+120]
            model_pred_test_175h = model_pred_test[:, :, :90+120]
            model_pred_test_2h = model_pred_test[:, :, :120*2]
            model_pred_test_225h = model_pred_test[:, :, :30+120*2]
            model_pred_test_250h = model_pred_test[:, :, :60+120*2]
            model_pred_test_275h = model_pred_test[:, :, :90+120*2]
            model_pred_test_3h = model_pred_test[:, :, :120*3]
            pred_test_025h = pred_test[:, :, :30]
            pred_test_050h = pred_test[:, :, :60]
            pred_test_075h = pred_test[:, :, :90]
            pred_test_1h = pred_test[:, :, :120]
            pred_test_125h = pred_test[:, :, :30+120]
            pred_test_150h = pred_test[:, :, :60+120]
            pred_test_175h = pred_test[:, :, :90+120]
            pred_test_2h = pred_test[:, :, :120*2]
            pred_test_225h = pred_test[:, :, :30+120*2]
            pred_test_250h = pred_test[:, :, :60+120*2]
            pred_test_275h = pred_test[:, :, :90+120*2]
            pred_test_3h = pred_test[:, :, :120*3]
            
            #===================#
            #===================#
            ade_err_km_sum_025h = 0
            for i in range(pred_test_025h.shape[-1]):
                d = get_distance_hav(pred_test_025h[:, 0, i].cpu().numpy(), pred_test_025h[:, 1, i].cpu().numpy(), model_pred_test_025h[:, 0, i].cpu().numpy(), model_pred_test_025h[:, 1, i].cpu().numpy())
            
                ade_err_km_sum_025h += d
            ade_err_km_avg_025h = ade_err_km_sum_025h/pred_test_025h.shape[-1]
            
            fde_err_km_025h = get_distance_hav(pred_test_025h[:, 0, -1].cpu().numpy(), pred_test_025h[:, 1, -1].cpu().numpy(), model_pred_test_025h[:, 0, -1].cpu().numpy(), model_pred_test_025h[:, 1, -1].cpu().numpy())
            
            #===================#
            #===================#
            ade_err_km_sum_050h = 0
            
            for i in range(pred_test_050h.shape[-1]):
                d = get_distance_hav(pred_test_050h[:, 0, i].cpu().numpy(), pred_test_050h[:, 1, i].cpu().numpy(), model_pred_test_050h[:, 0, i].cpu().numpy(), model_pred_test_050h[:, 1, i].cpu().numpy())
            
                ade_err_km_sum_050h += d
            ade_err_km_avg_050h = ade_err_km_sum_050h/pred_test_050h.shape[-1]
            
            fde_err_km_050h = get_distance_hav(pred_test_050h[:, 0, -1].cpu().numpy(), pred_test_050h[:, 1, -1].cpu().numpy(), model_pred_test_050h[:, 0, -1].cpu().numpy(), model_pred_test_050h[:, 1, -1].cpu().numpy())
            
            #===================#
            #===================#
            ade_err_km_sum_075h = 0
            for i in range(pred_test_075h.shape[-1]):
                d = get_distance_hav(pred_test_075h[:, 0, i].cpu().numpy(), pred_test_075h[:, 1, i].cpu().numpy(), model_pred_test_075h[:, 0, i].cpu().numpy(), model_pred_test_075h[:, 1, i].cpu().numpy())
            
                ade_err_km_sum_075h += d
            ade_err_km_avg_075h = ade_err_km_sum_075h/pred_test_075h.shape[-1]
            
            fde_err_km_075h = get_distance_hav(pred_test_075h[:, 0, -1].cpu().numpy(), pred_test_075h[:, 1, -1].cpu().numpy(), model_pred_test_075h[:, 0, -1].cpu().numpy(), model_pred_test_075h[:, 1, -1].cpu().numpy())
            
            #===================#
            #===================#
            ade_err_km_sum_1h = 0
            for i in range(pred_test_1h.shape[-1]):
                d = get_distance_hav(pred_test_1h[:, 0, i].cpu().numpy(), pred_test_1h[:, 1, i].cpu().numpy(), model_pred_test_1h[:, 0, i].cpu().numpy(), model_pred_test_1h[:, 1, i].cpu().numpy())
            
                ade_err_km_sum_1h += d
            ade_err_km_avg_1h = ade_err_km_sum_1h/pred_test_1h.shape[-1]
            
            fde_err_km_1h = get_distance_hav(pred_test_1h[:, 0, -1].cpu().numpy(), pred_test_1h[:, 1, -1].cpu().numpy(), model_pred_test_1h[:, 0, -1].cpu().numpy(), model_pred_test_1h[:, 1, -1].cpu().numpy())
            
            #===================#
            #===================#
            ade_err_km_sum_125h = 0
            for i in range(pred_test_125h.shape[-1]):
                d = get_distance_hav(pred_test_125h[:, 0, i].cpu().numpy(), pred_test_125h[:, 1, i].cpu().numpy(), model_pred_test_125h[:, 0, i].cpu().numpy(), model_pred_test_125h[:, 1, i].cpu().numpy())
            
                ade_err_km_sum_125h += d
            ade_err_km_avg_125h = ade_err_km_sum_125h/pred_test_125h.shape[-1]
            
            fde_err_km_125h = get_distance_hav(pred_test_125h[:, 0, -1].cpu().numpy(), pred_test_125h[:, 1, -1].cpu().numpy(), model_pred_test_125h[:, 0, -1].cpu().numpy(), model_pred_test_125h[:, 1, -1].cpu().numpy())
            
            #===================#
            #===================#
            ade_err_km_sum_150h = 0
            for i in range(pred_test_150h.shape[-1]):
                d = get_distance_hav(pred_test_150h[:, 0, i].cpu().numpy(), pred_test_150h[:, 1, i].cpu().numpy(), model_pred_test_150h[:, 0, i].cpu().numpy(), model_pred_test_150h[:, 1, i].cpu().numpy())
            
                ade_err_km_sum_150h += d
            ade_err_km_avg_150h = ade_err_km_sum_150h/pred_test_150h.shape[-1]
            
            fde_err_km_150h = get_distance_hav(pred_test_150h[:, 0, -1].cpu().numpy(), pred_test_150h[:, 1, -1].cpu().numpy(), model_pred_test_150h[:, 0, -1].cpu().numpy(), model_pred_test_150h[:, 1, -1].cpu().numpy())
            
            #===================#
            #===================#
            ade_err_km_sum_175h = 0
            for i in range(pred_test_175h.shape[-1]):
                d = get_distance_hav(pred_test_175h[:, 0, i].cpu().numpy(), pred_test_175h[:, 1, i].cpu().numpy(), model_pred_test_175h[:, 0, i].cpu().numpy(), model_pred_test_175h[:, 1, i].cpu().numpy())
            
                ade_err_km_sum_175h += d
            ade_err_km_avg_175h = ade_err_km_sum_175h/pred_test_175h.shape[-1]
            
            fde_err_km_175h = get_distance_hav(pred_test_175h[:, 0, -1].cpu().numpy(), pred_test_175h[:, 1, -1].cpu().numpy(), model_pred_test_175h[:, 0, -1].cpu().numpy(), model_pred_test_175h[:, 1, -1].cpu().numpy())
            
            #===================#
            #===================#
            ade_err_km_sum_2h = 0
            for i in range(pred_test_2h.shape[-1]):
                d = get_distance_hav(pred_test_2h[:, 0, i].cpu().numpy(), pred_test_2h[:, 1, i].cpu().numpy(), model_pred_test_2h[:, 0, i].cpu().numpy(), model_pred_test_2h[:, 1, i].cpu().numpy())
            
                ade_err_km_sum_2h += d
            ade_err_km_avg_2h = ade_err_km_sum_2h/pred_test_2h.shape[-1]
            
            fde_err_km_2h = get_distance_hav(pred_test_2h[:, 0, -1].cpu().numpy(), pred_test_2h[:, 1, -1].cpu().numpy(), model_pred_test_2h[:, 0, -1].cpu().numpy(), model_pred_test_2h[:, 1, -1].cpu().numpy())
            
            #===================#
            #===================#
            ade_err_km_sum_225h = 0
            for i in range(pred_test_225h.shape[-1]):
                d = get_distance_hav(pred_test_225h[:, 0, i].cpu().numpy(), pred_test_225h[:, 1, i].cpu().numpy(), model_pred_test_225h[:, 0, i].cpu().numpy(), model_pred_test_225h[:, 1, i].cpu().numpy())
            
                ade_err_km_sum_225h += d
            ade_err_km_avg_225h = ade_err_km_sum_225h/pred_test_225h.shape[-1]
            
            fde_err_km_225h = get_distance_hav(pred_test_225h[:, 0, -1].cpu().numpy(), pred_test_225h[:, 1, -1].cpu().numpy(), model_pred_test_225h[:, 0, -1].cpu().numpy(), model_pred_test_225h[:, 1, -1].cpu().numpy())
            
            #===================#
            #===================#
            ade_err_km_sum_250h = 0
            for i in range(pred_test_250h.shape[-1]):
                d = get_distance_hav(pred_test_250h[:, 0, i].cpu().numpy(), pred_test_250h[:, 1, i].cpu().numpy(), model_pred_test_250h[:, 0, i].cpu().numpy(), model_pred_test_250h[:, 1, i].cpu().numpy())
            
                ade_err_km_sum_250h += d
            ade_err_km_avg_250h = ade_err_km_sum_250h/pred_test_250h.shape[-1]
            
            fde_err_km_250h = get_distance_hav(pred_test_250h[:, 0, -1].cpu().numpy(), pred_test_250h[:, 1, -1].cpu().numpy(), model_pred_test_250h[:, 0, -1].cpu().numpy(), model_pred_test_250h[:, 1, -1].cpu().numpy())
            
            #===================#
            #===================#
            ade_err_km_sum_275h = 0
            for i in range(pred_test_275h.shape[-1]):
                d = get_distance_hav(pred_test_275h[:, 0, i].cpu().numpy(), pred_test_275h[:, 1, i].cpu().numpy(), model_pred_test_275h[:, 0, i].cpu().numpy(), model_pred_test_275h[:, 1, i].cpu().numpy())
            
                ade_err_km_sum_275h += d
            ade_err_km_avg_275h = ade_err_km_sum_275h/pred_test_275h.shape[-1]
            
            fde_err_km_275h = get_distance_hav(pred_test_275h[:, 0, -1].cpu().numpy(), pred_test_275h[:, 1, -1].cpu().numpy(), model_pred_test_275h[:, 0, -1].cpu().numpy(), model_pred_test_275h[:, 1, -1].cpu().numpy())
            
            #===================#
            #===================#
            ade_err_km_sum_3h = 0
            for i in range(pred_test_3h.shape[-1]):
                d = get_distance_hav(pred_test_3h[:, 0, i].cpu().numpy(), pred_test_3h[:, 1, i].cpu().numpy(), model_pred_test_3h[:, 0, i].cpu().numpy(), model_pred_test_3h[:, 1, i].cpu().numpy())
            
                ade_err_km_sum_3h += d
            ade_err_km_avg_3h = ade_err_km_sum_3h/pred_test_3h.shape[-1]
            
            fde_err_km_3h = get_distance_hav(pred_test_3h[:, 0, -1].cpu().numpy(), pred_test_3h[:, 1, -1].cpu().numpy(), model_pred_test_3h[:, 0, -1].cpu().numpy(), model_pred_test_3h[:, 1, -1].cpu().numpy())
            #===================#
            #===================#
            
            test_results['ADE_avg_025h'] += ade_err_km_avg_025h
            test_results['ADE_avg_050h'] += ade_err_km_avg_050h
            test_results['ADE_avg_075h'] += ade_err_km_avg_075h
            test_results['ADE_avg_1h'] += ade_err_km_avg_1h
            test_results['ADE_avg_125h'] += ade_err_km_avg_125h
            test_results['ADE_avg_150h'] += ade_err_km_avg_150h
            test_results['ADE_avg_175h'] += ade_err_km_avg_175h
            test_results['ADE_avg_2h'] += ade_err_km_avg_2h
            test_results['ADE_avg_225h'] += ade_err_km_avg_225h
            test_results['ADE_avg_250h'] += ade_err_km_avg_250h
            test_results['ADE_avg_275h'] += ade_err_km_avg_275h
            test_results['ADE_avg_3h'] += ade_err_km_avg_3h
            test_results['FDE_025h'] += fde_err_km_025h
            test_results['FDE_050h'] += fde_err_km_050h
            test_results['FDE_075h'] += fde_err_km_075h
            test_results['FDE_1h'] += fde_err_km_1h
            test_results['FDE_125h'] += fde_err_km_125h
            test_results['FDE_150h'] += fde_err_km_150h
            test_results['FDE_175h'] += fde_err_km_175h
            test_results['FDE_2h'] += fde_err_km_2h
            test_results['FDE_225h'] += fde_err_km_225h
            test_results['FDE_250h'] += fde_err_km_250h
            test_results['FDE_275h'] += fde_err_km_275h
            test_results['FDE_3h'] += fde_err_km_3h
            
            if ade_err_km_avg_1h > 500:
                print('obs_org_test', obs_org_test)
                print('pred_org_test', pred_org_test)
                print('model_pred_test', model_pred_test)
                print('pred_test', pred_test)
                
            test_bar.set_description(desc="ADE 1h: %.2f 2h: %.2f 3h: %.2f FDE 1h: %.2f 2h: %.2f 3h: %.2f" % (ade_err_km_avg_1h, ade_err_km_avg_2h, ade_err_km_avg_3h, fde_err_km_1h, fde_err_km_2h, fde_err_km_3h)) 
   
    print("*********************")
    print('Results are in km')
    print("mean 025h_ADE_avg:{}".format(test_results['ADE_avg_025h']/len(test_loader)))
    print("mean 050h_ADE_avg:{}".format(test_results['ADE_avg_050h']/len(test_loader)))
    print("mean 075h_ADE_avg:{}".format(test_results['ADE_avg_075h']/len(test_loader)))
    print("mean 1h_ADE_avg:{}".format(test_results['ADE_avg_1h']/len(test_loader)))
    print("mean 125h_ADE_avg:{}".format(test_results['ADE_avg_125h']/len(test_loader)))
    print("mean 150h_ADE_avg:{}".format(test_results['ADE_avg_150h']/len(test_loader)))
    print("mean 175h_ADE_avg:{}".format(test_results['ADE_avg_175h']/len(test_loader)))
    print("mean 2h_ADE_avg:{}".format(test_results['ADE_avg_2h']/len(test_loader)))
    print("mean 225h_ADE_avg:{}".format(test_results['ADE_avg_225h']/len(test_loader)))
    print("mean 250h_ADE_avg:{}".format(test_results['ADE_avg_250h']/len(test_loader)))
    print("mean 275h_ADE_avg:{}".format(test_results['ADE_avg_275h']/len(test_loader)))
    print("mean 3h_ADE_avg:{}".format(test_results['ADE_avg_3h']/len(test_loader)))
    print("mean 025h_FDE:{}".format(test_results['FDE_025h']/len(test_loader)))
    print("mean 050h_FDE:{}".format(test_results['FDE_050h']/len(test_loader)))
    print("mean 075h_FDE:{}".format(test_results['FDE_075h']/len(test_loader)))
    print("mean 1h_FDE:{}".format(test_results['FDE_1h']/len(test_loader)))
    print("mean 125h_FDE:{}".format(test_results['FDE_125h']/len(test_loader)))
    print("mean 150h_FDE:{}".format(test_results['FDE_150h']/len(test_loader)))
    print("mean 175h_FDE:{}".format(test_results['FDE_175h']/len(test_loader)))
    print("mean 2h_FDE:{}".format(test_results['FDE_2h']/len(test_loader)))
    print("mean 225h_FDE:{}".format(test_results['FDE_225h']/len(test_loader)))
    print("mean 250h_FDE:{}".format(test_results['FDE_250h']/len(test_loader)))
    print("mean 275h_FDE:{}".format(test_results['FDE_275h']/len(test_loader)))
    print("mean 3h_FDE:{}".format(test_results['FDE_3h']/len(test_loader)))
    print("Inference time: ", sum(inf_time)/(len(test_bar)*batch_size))
if __name__ == "__main__":
    main(draw = False)
