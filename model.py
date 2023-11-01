import math
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors
import numpy as np
import time
import matplotlib.pyplot as plt
from torch.nn.utils import weight_norm
from dtaidistance import dtw

class Chomp1d_anti(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d_anti, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x): 
        return x[:, :, self.chomp_size:].contiguous()    #anti_tcn
        #return x[:, :, :-self.chomp_size].contiguous()  #org_tcn

class Chomp1d_casual(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d_casual, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x): 
        #return x[:, :, self.chomp_size:].contiguous()    #anti_tcn
        return x[:, :, :-self.chomp_size].contiguous()  #org_tcn

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, padding_casual, dropout=0):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.conv12 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding_casual, dilation=dilation))
        self.chomp1 = Chomp1d_anti(padding)
        self.chomp12 = Chomp1d_casual(padding_casual)
        self.relu1 = nn.ELU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.conv22 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                   stride=stride, padding=padding_casual, dilation=dilation))
        self.chomp2 = Chomp1d_anti(padding)
        self.chomp22 = Chomp1d_casual(padding_casual)
       
        self.relu2 = nn.ELU()
        self.dropout2 = nn.Dropout(dropout)

        self.net_anti = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.net_casual = nn.Sequential(self.conv12, self.chomp12, self.relu1, self.dropout1,
                                 self.conv22, self.chomp22, self.relu2, self.dropout2)
        #self.net = nn.Sequential(self.conv1, self.chomp1, self.dropout1,
                                 #self.conv2, self.chomp2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.upsample = nn.Linear(n_outputs*2, n_outputs)
        self.relu = nn.ELU()
        self.init_weights()
        self.n_inputs = n_outputs
        
    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        self.conv12.weight.data.normal_(0, 0.01)
        self.conv22.weight.data.normal_(0, 0.01)
        if self.downsample is not None:        
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out_anti = self.net_anti(x)
        out_casual = self.net_casual(x)
        out = torch.cat((out_anti, out_casual), dim = 1)
        out = self.upsample(out.permute(0,2,1)).permute(0,2,1)
        res = x if self.downsample is None else self.downsample(x)
        out = out+res
        
        
        #return self.relu(out + res)
        return out 


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, padding_casual=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.padding_casual=(kernel_size-1) * dilation_size
        self.padding=(kernel_size-1) * dilation_size
        self.dilation = dilation_size
        self.network = nn.Sequential(*layers)
        self.kernel = kernel_size
    def forward(self, x):
        return self.network(x)

class SelfAttention(nn.Module):
    def __init__(self, num, channels, length, c_out):
        super(SelfAttention, self).__init__()
        self.num = num
        self.channels = channels
        self.len = length
        self.c_out = c_out
        self.mha = nn.MultiheadAttention(channels, 1, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, c_out),
        )
        self.fc = nn.Sequential(
            nn.Linear(num*length, length),
        )

    def forward(self, x):
        # [5, 2, 480] / [1, 2, 2400]
        x = x.permute(0, 2, 1) # [5, 480, 2] / [1, 2400, 2]
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        
        attention_value = self.ff_self(attention_value) + attention_value 
        return self.fc(attention_value.permute(0,2,1)).permute(2,1,0)  # [480, 2, 5] -> [2, 1, 0] / [1, 2, 2400]


    
class Encoder(nn.Module):
    def __init__(self, c_in=2, obs_in = 120, pred_in = 120, db_in = 240, top_num = 5, tcn_hidden = [2,2,2], device="cuda"):
        super().__init__()
        self.device = device
        self.top_num = top_num
        self.ma = SelfAttention(top_num + 1, tcn_hidden[-1], db_in, c_in)
        self.atcn_top = TemporalConvNet(num_inputs = c_in, num_channels = tcn_hidden)
        self.lstm_db = nn.LSTM(c_in, c_in, batch_first = True, bidirectional = True)

        
        self.fc = nn.Sequential(
            nn.Linear(obs_in, obs_in*4),
            nn.LeakyReLU(0.1),
            nn.Linear(obs_in*4, obs_in),
        )
        
        self.fc_top = nn.Sequential(
            nn.Linear(db_in, db_in*4),
            nn.LeakyReLU(0.1),
            nn.Linear(db_in*4, db_in),
        )

        
    def find_similar(self, obs, db_org):
        dist = []
        
        for i in range(len(db_org)):
            mse = dtw.distance_fast(db_org[0][:, :, :obs.shape[-1]].detach().clone().cpu().numpy().squeeze(axis = 0).flatten(), db_org[i][:, :, :obs.shape[-1]].detach().clone().cpu().numpy().squeeze(axis = 0).flatten())
            #mse = mean_squared_error(db_org[0][:, :, :obs.shape[-1]].detach().clone().cpu().numpy().squeeze(axis = 0).flatten(), db_org[i][:, :, :obs.shape[-1]].detach().clone().cpu().numpy().squeeze(axis = 0).flatten())
            
            dist.append(mse)
                
        top_sim_id_ = np.argsort(dist)
        
        top_sim_id_ = top_sim_id_[:min(self.top_num+1, len(db_org))].tolist()
        
        top_sim_id = []
        for i in range(len(top_sim_id_)):
            if dist[top_sim_id_[i]] < 1:
                top_sim_id.append(top_sim_id_[i])
           
        top_org = [db_org[i] for i in top_sim_id]

        top_zeros_org = torch.zeros_like(db_org[0]).cuda()
        if len(top_sim_id) < (self.top_num+1):
            lack = (self.top_num+1) - len(top_sim_id)
            for i in range(lack):
                top_org.append(top_zeros_org)
                
        top_sim_org = torch.stack(top_org, axis = 1).cuda()  
        return 0, top_sim_org.squeeze(dim = 0).permute(1,0,2)
    
    
    def forward(self, obs, pred, db):
        obs_fill = torch.zeros_like(pred).cuda()
        obs_org = torch.cat((obs.cuda(), obs_fill), dim = -1)
        db_org = [obs_org]
        if self.top_num != 0:
            for i in range(len(db)):
                db_org.append(db[i].cuda())

        top_sim_emb, top_sim_org = self.find_similar(obs, db_org) # top_sim_org [2, 5, 480]
        top_sim_org = torch.flatten(top_sim_org, 1).unsqueeze(dim = 0).float() # [1, 2, 2400]
        top_sim_org = self.atcn_top(x = top_sim_org)
        top_sim_org = self.ma(top_sim_org.float())
        top_sim_org = top_sim_org.permute(2,1,0)
        
        return top_sim_emb, top_sim_org


