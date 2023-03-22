import math,os
import json
from PIL import Image, ImageDraw
from collections import Counter
import torch
import random
import numpy as np
from torchvision.transforms import Compose, ToTensor, Resize
import torchvision.transforms as T

def get_rank(score_list,index):
    score = score_list[index]
    arr = np.sort(score_list)
    rank = np.where(arr==score)
    return rank[0][-1]

def lonlat2meters(lon, lat):
    semimajoraxis = 6378137.0
    east = lon * 0.017453292519943295
    north = lat * 0.017453292519943295
    t = math.sin(north)
    return semimajoraxis * east, 3189068.5 * math.log((1 + t) / (1 - t))

region = {}.fromkeys(['minlat','maxlat','minlon','maxlon','cellsize','numx','numy'])

with open('edit_hyper_parameters.json') as json_file:
    parameters = json.load(json_file)
    region['minlon'], region['minlat'] = lonlat2meters(parameters['region']['minlon'], parameters['region']['minlat'])
    region['maxlon'], region['maxlat'] = lonlat2meters(parameters['region']['maxlon'], parameters['region']['maxlat'])
    region['cellsize'] = parameters['region']['cellsize']
    region['cellsize_lr'] = parameters['region']['cellsize_lr']
    region['imgsize_x'] = parameters['region']['imgsize_x']
    region['imgsize_y'] = parameters['region']['imgsize_y']
    region['imgsize_x_lr'] = parameters['region']['imgsize_x_lr']
    region['imgsize_y_lr'] = parameters['region']['imgsize_y_lr']
    region['numy'] = int(round(region['maxlat'] - region['minlat'], 6) / region['cellsize'])
    region['numx_lr'] = int(round(region['maxlon'] - region['minlon'], 6) / region['cellsize_lr']) 
    region['numy_lr'] = int(round(region['maxlat'] - region['minlat'], 6) / region['cellsize_lr']) 
    region['pixelrange'] = parameters['region']['pixelrange']
    region['pixelrange_lr'] = parameters['region']['pixelrange_lr']

def coord2cell(x,y):
    xoffset = (x - region['minlon'])/(region['maxlon']-region['minlon'])* region['imgsize_x']/region['pixelrange']
    yoffset = (region['maxlat'] - y)/(region['maxlat']-region['minlat'])* region['imgsize_y']/region['pixelrange']
    xoffset = int(xoffset)
    yoffset = int(yoffset)
    tmp = (region['imgsize_x'])/(region['pixelrange'])
    id = yoffset * tmp + xoffset
    return id

def coord2cell_lr(x, y):    
    xoffset = (x - region['minlon']) / (region['maxlon'] - region['minlon']) * region['imgsize_x_lr'] / region['pixelrange_lr']
    yoffset = (region['maxlat'] - y) / (region['maxlat'] - region['minlat']) * region['imgsize_y_lr'] / region['pixelrange_lr']
    xoffset = int(xoffset)
    yoffset = int(yoffset)
    tmp = region['imgsize_x_lr'] / (region['pixelrange_lr'])
    id = yoffset * tmp + xoffset 
    return id
'''
find the anchor points of the cell
'''
def cell2anchor(xoffset, yoffset, pixel):
    left_upper_point_x = xoffset * pixel
    left_upper_point_y = yoffset * pixel
    right_lower_point_x = left_upper_point_x + pixel - 1
    right_lower_point_y = left_upper_point_y + pixel - 1
    return (left_upper_point_x, left_upper_point_y), (right_lower_point_x, right_lower_point_y)

def draw(seq, index, mode):
    img = Image.new("L", (region['imgsize_x'],region['imgsize_y']))
    cellset = Counter(seq).keys() 
    occurrence = Counter(seq).values() 
    for i, cell in enumerate(cellset):
        xoffset = cell % (region['imgsize_x']/region['pixelrange'])
        yoffset = (cell // (region['imgsize_y']/(region['pixelrange']))) % (region['imgsize_x']/region['pixelrange'])
        left_upper_point, right_lower_point = cell2anchor(xoffset, yoffset, region['pixelrange'])
        grayscale = 64 + list(occurrence)[i]*10 if list(occurrence)[i] < 20 else 255
        shape = [left_upper_point, right_lower_point]
        ImageDraw.Draw(img).rectangle(shape, fill=(grayscale))
    return img

def draw_pred(seq, index, mode):
    img = Image.new("L", (region['imgsize_x'],region['imgsize_y']))
    cellset = Counter(seq).keys() 
    occurrence = Counter(seq).values() 
    for i, cell in enumerate(cellset):

        xoffset = cell % (region['imgsize_x']/region['pixelrange'])
        yoffset = (cell // (region['imgsize_y']/(region['pixelrange']))) % (region['imgsize_x']/region['pixelrange'])

        left_upper_point, right_lower_point = cell2anchor(xoffset, yoffset, region['pixelrange'])
        grayscale = 64 + list(occurrence)[i]*10 if list(occurrence)[i] < 20 else 255
        shape = [left_upper_point, right_lower_point]
        ImageDraw.Draw(img).rectangle(shape, fill=(grayscale))
    return img

def draw_db(seq, index):
    img = Image.new("L", (region['imgsize_x'],region['imgsize_y']))
    cellset = Counter(seq).keys() 
    occurrence = Counter(seq).values() 
    for i, cell in enumerate(cellset):

        xoffset = cell % (region['imgsize_x']/region['pixelrange'])
        yoffset = (cell // (region['imgsize_y']/(region['pixelrange']))) % (region['imgsize_x']/region['pixelrange'])

        left_upper_point, right_lower_point = cell2anchor(xoffset, yoffset, region['pixelrange'])
        grayscale = 64 + list(occurrence)[i]*10 if list(occurrence)[i] < 20 else 255
        shape = [left_upper_point, right_lower_point]
        ImageDraw.Draw(img).rectangle(shape, fill=(grayscale))
    return img

def draw_lr(seq, index, mode):
    
    img = Image.new("L", (region['imgsize_x_lr'],region['imgsize_y_lr']))
    cellset = Counter(seq).keys() 
    occurrence = Counter(seq).values() 
    for i, cell in enumerate(cellset):
        xoffset = cell % (region['imgsize_x_lr']/region['pixelrange_lr'])
        yoffset = (cell // (region['imgsize_y_lr']/region['pixelrange_lr'])) % (region['imgsize_x_lr']/region['pixelrange_lr'])
        left_upper_point, right_lower_point = cell2anchor(xoffset, yoffset, region['pixelrange_lr'])
        grayscale = 64 + list(occurrence)[i]*10 if list(occurrence)[i] < 20 else 255
        shape = [left_upper_point, right_lower_point]
        ImageDraw.Draw(img).rectangle(shape, fill=(grayscale))
    return img

def traj2cell(seq):
    cell_seq = []
    for j in range(seq.shape[1]):
        x, y = lonlat2meters(seq[0][j], seq[1][j])
        cell_seq.append(coord2cell(x, y)) 
    return cell_seq

def traj2cell_lr(seq):
    cell_seq = []
    for j in range(seq.shape[1]):
        x, y = lonlat2meters(seq[0][j], seq[1][j]) 
        cell_seq.append(coord2cell_lr(x, y)) 
    return cell_seq

def traj2cell_test(seq):
    cell_seq = []
    for j in range(seq.shape[0]): 
        x, y = lonlat2meters(seq[j][0], seq[j][1])  
        cell_seq.append(coord2cell(x, y))
    return cell_seq


def traj2cell_test_lr(seq):
    cell_seq = []
    for j in range(seq.shape[0]):
        x, y = lonlat2meters(seq[j][0], seq[j][1]) 
        cell_seq.append(coord2cell_lr(x, y)) 
    return cell_seq


with open('edit_hyper_parameters.json') as json_file:
    parameters = json.load(json_file)
    minlon = parameters['region']['minlon']
    minlat = parameters['region']['minlat']
    maxlon = parameters['region']['maxlon']
    maxlat = parameters['region']['maxlat']
    json_file.close()


def create_dataset(traj, index, mode):
    traj = traj.transpose() 
    traj_len = traj.shape[0] 
    rand_start = random.randint(0, traj_len - 90) 
    traj_crop = traj[rand_start : rand_start + 90, :]
    
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

    lr = traj[(traj[:, 0] < lr_latmax) & (traj[:, 0] > lr_latmin) & (traj[:, 1] < lr_lonmax) & (traj[:, 1] > lr_lonmin)]
    hr = traj[(traj[:, 0] < hr_latmax) & (traj[:, 0] > hr_latmin) & (traj[:, 1] < hr_lonmax) & (traj[:, 1] > hr_lonmin)] 
    
    lr = lr.transpose()
    hr = hr.transpose()
    
    origin = traj2cell(hr)
    hr_img = ToTensor()(draw(origin, index, mode+'_hr'))
    
    lr = traj2cell(lr)
    lr_img = ToTensor()(draw(lr, index, mode+'_lr'))
    
    if not os.path.exists("data/imagedata/dma_src_train/"):
        os.makedirs("data/imagedata/dma_src_train/")
    if not os.path.exists("data/imagedata/dma_trg_train/"):
        os.makedirs("data/imagedata/dma_trg_train/")
    if not os.path.exists("data/imagedata/dma_src_val/"):
        os.makedirs("data/imagedata/dma_src_val/")
    if not os.path.exists("data/imagedata/dma_trg_val/"):
        os.makedirs("data/imagedata/dma_trg_val/")
    
    torch.save(lr_img, "data/imagedata/dma_src_{}/{}.data".format(mode, index))
    torch.save(hr_img, "data/imagedata/dma_trg_{}/{}.data".format(mode, index))