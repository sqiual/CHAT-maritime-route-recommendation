import argparse, os, shutil
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from trajsim_model import Encoder
from dataset import MyDataset
from tqdm import tqdm
import pytorch_ssim
import csv
from torchvision.transforms import ToTensor, ToPILImage
import torchvision.transforms as T
from utils import *
import time 
import torch.nn.functional as F


parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--upscale_factor', default=2, type=int, choices=[2],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')
parser.add_argument('--start_epoch', default=0, type=int, help='start epoch number')
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--lrD", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--lrG", type=float, default=1e-5, help="Learning Rate. Default=2e-4")
parser.add_argument("--pretrained", default='checkpoint/simdma_bestmodel_MyG_3.pt', type=str, help="path to pretrained model (default: none)")
#parser.add_argument("--pretrained", default=None, type=str, help="path to pretrained model (default: none)")
parser.add_argument("--print_freq", default=1, type=int, help="the freq of print during training")
parser.add_argument("--save_freq", default=1, type=int, help="the freq of save checkpoint")
parser.add_argument("--batch_size", default=2, type=int, help="batch size")


def main():
    opt = parser.parse_args()
    print(opt)
    train_set = MyDataset('data/imagedata/dma_trg_train', 'data/imagedata/dma_src_train','train')
    val_set = MyDataset('data/imagedata/dma_trg_val', 'data/imagedata/dma_src_val','val')
    train_loader = DataLoader(dataset=train_set, batch_size=opt.batch_size, num_workers=4, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=opt.batch_size, num_workers=4, shuffle=False)

    if not torch.cuda.is_available():
        raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)
    cudnn.benchmark = True

    print("===> Setting GPU")
    
    netD = Encoder()
    gname = "MyG_3"
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))
    criterion = nn.MSELoss(reduction="sum") 
    criterion_mse = nn.MSELoss()
    criterion_e = nn.L1Loss(reduction="sum") 
    

    netD.cuda()
    criterion.cuda()
    criterion_mse.cuda()
    criterion_e.cuda()

    print("===> Setting Optimizer")
    optimizerD = optim.Adam(netD.parameters(),lr=opt.lrD)

    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading checkpoint '{}'".format(opt.pretrained))
            checkpoint = torch.load(opt.pretrained)
            opt.start_epoch = checkpoint["epoch"]
            best_loss = checkpoint["best_loss"]
            netD.load_state_dict(checkpoint["netD"])
            optimizerD.load_state_dict(checkpoint["optimizerD"])
        else:
            print("=> no model found at '{}'".format(opt.pretrained))
    else:
        best_loss = float('inf')
        best_vec_loss = float('inf')

    results = {'d_loss':[], 'vec_mse':[]}
    
    
    for epoch in range(opt.start_epoch + 1, opt.num_epochs + opt.start_epoch + 1):
        train_bar = tqdm(train_loader)

        netD.train()

        for lr, hr in train_bar:
            lr, hr = Variable(lr, requires_grad=True), Variable(hr, requires_grad=False)
            lr = lr.cuda()
            hr = hr.cuda()
            
            ############################
            # (1) Update embedding block:
            ###########################
            netD.zero_grad()
            real_out = netD(hr)
            fake_out = netD(lr)
            d_loss = criterion_e(real_out, fake_out)
            d_loss.backward()
            optimizerD.step()
            
            train_bar.set_description(desc='[%d/%d] d_loss: %.3f' % (
                epoch, opt.num_epochs + opt.start_epoch, d_loss.item()))

        netD.eval()
        out_path = 'training_results/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        with torch.no_grad():
            valing_results = {'vec_mse': 0}
            val_bar = tqdm(val_loader)
            for j, batch in enumerate(val_bar):
                val_lr, val_hr = Variable(batch[0]), Variable(batch[1], requires_grad=False)
                val_lr = val_lr.cuda()
                val_hr = val_hr.cuda()
                
                
                real_out = netD(val_hr)
                fake_out = netD(val_lr)
                vec_mse = criterion_e(real_out, fake_out)

                valing_results['vec_mse'] += vec_mse.item()
                val_lr = val_lr.to(torch.device("cpu"))
                val_hr = val_hr.to(torch.device("cpu"))
                train_bar.set_description(desc="validating……")

        results['d_loss'].append(d_loss.item())
        results['vec_mse'].append(valing_results['vec_mse']/len(val_loader))

        if epoch % opt.print_freq == 0:
            print("epoch: {}\tvector_loss:{} ".format(epoch, valing_results['vec_mse'] / len(val_loader)))
        if epoch % opt.save_freq == 0:
            vec_loss = valing_results['vec_mse']/len(val_loader)

            if vec_loss  < best_loss:
                best_loss = vec_loss
                is_best = True
            else:
                is_best = False
            save_checkpoint({
                "epoch": epoch,
                "best_loss": best_loss,
                "netD": netD.state_dict(),
                "optimizerD": optimizerD.state_dict()
            }, is_best, gname)


def save_checkpoint(state, is_best, name):
    filename = "checkpoint/simdma_checkpoint_%s_epoch_%d.pt" % (name,state["epoch"])
    if is_best:
        print("saving the epoch {} as best model".format(state["epoch"]))
        torch.save(state, filename)
        shutil.copyfile(filename, 'checkpoint/simdma_bestmodel_%s.pt'%name)


if __name__ == "__main__":
    main()
