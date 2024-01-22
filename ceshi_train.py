import os
import argparse, shutil
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import h5py
import numpy as np
import math
from dataset import MyDataset
from tqdm.auto import tqdm
import csv
import json
from pathlib import Path
import pandas as pd
import random 
import pytorch_ssim
from torch.autograd import Variable
import torch
from model import Encoder, TemporalConvNet
import time

parser = argparse.ArgumentParser(description='Train Trajectory Prediction Models')
parser.add_argument('--num_epoch', default=10, type=int, help='train epoch number')
parser.add_argument('--batch_size', default=1, type=int, help='train epoch number')
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")

parser.add_argument('--c_in', default=2, type=int, help='number of features, lat, lon')
parser.add_argument('--obs_len', default=120, type=int, help='length of the observed trajeatory')
parser.add_argument('--pred_len', default=360, type=int, help='length of the predicted trajeatory')
parser.add_argument('--db_len', default=480, type=int, help='length of the whole trajeatory')
parser.add_argument('--step', default=500, type=int, help='# of step for diffusion')
parser.add_argument('--tcn_hidden', default=[2,2,2,2,2], type=list, help='# of step for diffusion')

#=====================
parser.add_argument("--top", type=int, default=7, help="top n similar trajectory used for prediction")
parser.add_argument('--best_model', default='checkpoint/CHAT_star6_BTCN_top7_bestmodel.pt', help='name of the best model')
#parser.add_argument("--pretrained", default='checkpoint/CHAT_star6_BTCN_15day_top5_bestmodel_new1.pt', type=str, help="path to pretrained model (default: none)")  
parser.add_argument("--pretrained", default=None, type=str, help="path to pretrained model (default: none)")
#=====================


def save_checkpoint(state, is_best, modelname):
    filename = "checkpoint/CHAT_star6_BTCN_epoch_%d.pt" % (state["epoch"])
    if is_best:
        print("##### saving epoch {} as the best model #####".format(state["epoch"]))
        torch.save(state, filename)
        shutil.copyfile(filename, modelname) 


def main():
    args = parser.parse_args()
    print(args)
    train_set = MyDataset('data/processed_data/dma_src_train', 'data/processed_data/dma_trg_train', 'data/processed_data/dma_whole')
    val_set = MyDataset('data/processed_data/dma_src_val', 'data/processed_data/dma_trg_val', 'data/processed_data/dma_whole')
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, num_workers=1, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size, num_workers=1, shuffle=False)

    # ==============================
    #         set things up
    # ==============================

    seed = random.randint(1, 10000)
    #print("Random Seed: ", seed)
    torch.manual_seed(seed)
    
    netE = Encoder(c_in=args.c_in, obs_in = args.obs_len, pred_in = args.pred_len, db_in = args.db_len, top_num = args.top, tcn_hidden = args.tcn_hidden).cuda()
    optimizerE = optim.Adam(netE.parameters(), args.lr)
    schedulerE = optim.lr_scheduler.MultiStepLR(optimizerE, milestones=[80, 160], gamma=0.5)
    
    #netP = TemporalConvNet(num_inputs = args.c_in, num_channels = args.tcn_hidden).cuda()
    #optimizerP = optim.Adam(netP.parameters(), args.lr)
    #schedulerP = optim.lr_scheduler.MultiStepLR(optimizerP, milestones=[200, 300], gamma=0.5)
    
    criterion_l1 = nn.L1Loss(reduction="sum")
    criterion_l1.cuda()
    criterion_de = nn.MSELoss()
    criterion_de.cuda()
    criterion_mse = nn.MSELoss(reduction="sum")
    criterion_mse.cuda()
    
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained)
            pred_start_epoch = checkpoint["epoch"]
            val_best_loss = checkpoint["val_best_loss"]
            netE.load_state_dict(checkpoint["Encoder"])
            optimizerE.load_state_dict(checkpoint["optimizerE"])
            #netP.load_state_dict(checkpoint["Prediction"])
            #optimizerP.load_state_dict(checkpoint["optimizerP"])
        else:
            print("=> no model found at '{}'".format(args.pretrained))
    else:
        val_best_loss = float('inf')
        pred_start_epoch = 0
    
    print('Training begins ...')
    ##################################################
    # Start training 
    ##################################################

    for epoch in range(pred_start_epoch, args.num_epoch):
        # ------------Training---------------
        #netP.train()
        netE.train()
        train_bar = tqdm(train_loader)

        for num, data in enumerate(train_bar):
            obs, pred, db, origin = data  # [N, 2, obs_len], [N, 2, pred_len], # of traj [#, 2, obs_pred_len] [1, 2]
            whole = torch.cat((obs, pred), dim = -1).cuda()  # [N, 2, obs_len+pred_len]
            #---------------------------------------
            # Begin Encoding
            top_sim_emb, top_sim_org = netE(obs, pred, db) # [1,32,240] [1,32,240] [1, 2, 240] 
            #---------------------------------------
            # Make prediction 
            
            #model_pred = netP(x = top_sim_org)
            model_pred = top_sim_org
            encode_loss = criterion_l1(whole[:, :, args.obs_len:].to(torch.float64), top_sim_org[:, :, args.obs_len:].to(torch.float64))
            #pred_loss = criterion_l1(model_pred[:, :, args.obs_len:].to(torch.float64), pred.to(torch.float64).cuda())
            ##############################
            # Update Prediction block:
            ##############################
            #netP.zero_grad()
            #pred_loss.backward(retain_graph=True)
            #optimizerP.step()
            
            ############################
            # Update embedding block:
            ###########################
            netE.zero_grad()
            encode_loss.backward()
            optimizerE.step()
            
            train_bar.set_description(desc='[%d/%d] Training encode_loss: %.4f' % (epoch, args.num_epoch, encode_loss.item()))
        schedulerE.step()
        
        # ------------Evaluation---------------
        netE.eval()
        #netP.eval()
        with torch.no_grad():
            valing_results = {'e_loss': 0, 'p_loss': 0}
            val_bar = tqdm(val_loader)
            for num, data in enumerate(val_bar):
                obs_val, pred_val, db_val, origin = data  # [N, 2, obs_len], [N, 2, pred_len], # of traj [#, 2, obs_pred_len]
                
                whole_val = torch.cat((obs_val, pred_val), dim = -1) # [N, 2, obs_len+pred_len]
                #---------------------------------------
                # Begin Encoding
                top_sim_emb_val, top_sim_org_val = netE(obs_val, pred_val, db_val)

                #---------------------------------------
                # Make prediction  
                #model_pred_val = netP(x = top_sim_org_val)
                #model_pred_val = top_sim_org_val
                #pred_loss_val = criterion_l1(model_pred_val[:, :, args.obs_len:].to(torch.float64), pred_val.to(torch.float64).cuda())
                encode_loss_val = criterion_l1(whole_val[:, :, args.obs_len:].cuda(), top_sim_org_val[:, :, args.obs_len:].cuda())
                
                valing_results['e_loss'] += encode_loss_val.item()
                #valing_results['p_loss'] += pred_loss_val.item()
                val_bar.set_description(desc="Validating e_loss: %.4f" % (encode_loss_val.item())) 
            
        batch_val_loss = valing_results['e_loss']/len(val_loader)
        
        print("epoch:{} batch mean e_loss:{}".format(epoch, batch_val_loss/args.batch_size))
        if batch_val_loss < val_best_loss:
            val_best_loss = batch_val_loss
            is_best = True

        else:
            is_best = False

        save_checkpoint({
            "epoch": epoch,
            "val_best_loss": val_best_loss,
            "Encoder": netE.state_dict(),
            "optimizerE": optimizerE.state_dict(),
            #"Prediction": netP.state_dict(),
            #"optimizerP": optimizerP.state_dict()
        }, is_best, args.best_model)
                
if __name__ == "__main__":
    main()
            
