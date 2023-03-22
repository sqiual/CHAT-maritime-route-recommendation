import argparse, os, shutil
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
# from model import MyDiscriminator,MyGenerator
from diffusion_model import *
from diffusion_modules import UNet, sr_images
from dataset import MyDataset
from tqdm import tqdm
# from torchsummary import summary
import pytorch_ssim
import csv
# import math
from torchvision.transforms import ToTensor, ToPILImage
import torchvision.transforms as T
from utils import *
import time 


parser = argparse.ArgumentParser(description='Train Super Resolution Models')
# parser.add_argument('--crop_size', default=88, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=2, type=int, choices=[2],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')
parser.add_argument('--start_epoch', default=0, type=int, help='start epoch number')
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--lrD", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--lrG", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--pretrained", default='checkpoint/dma_bestmodel_MyG_3.pt', type=str, help="path to pretrained model (default: none)")
# parser.add_argument("--pretrained", default=None, type=str, help="path to pretrained model (default: none)")
parser.add_argument("--print_freq", default=1, type=int, help="the freq of print during training")
parser.add_argument("--save_freq", default=1, type=int, help="the freq of save checkpoint")
parser.add_argument("--batch_size", default=4, type=int, help="batch size")


########################################################
# 要记得去 dataset 里的 MyDataset 里去改 amount
########################################################


def main():
    opt = parser.parse_args()
    print(opt)
    train_set = MyDataset('autodl-nas/trajSR/image/dma_trg_train', 'autodl-nas/trajSR/image/dma_src_train','train')
    val_set = MyDataset('autodl-nas/trajSR/image/dma_trg_val', 'autodl-nas/trajSR/image/dma_src_val','val')
    # 由于报OOM3，batch size由原来的32改成4
    train_loader = DataLoader(dataset=train_set, batch_size=opt.batch_size, num_workers=8, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=opt.batch_size, num_workers=8, shuffle=False)

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if not torch.cuda.is_available():
        raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)
    cudnn.benchmark = True

    print("===> Setting GPU")
    
    # 拿进去的是lr的图片
    netG = Diffusion(noise_steps=100, beta_start=-6, beta_end=6, img_size=256, device="cuda")
    diffu_model = UNet().cuda()
    # 这里要改netG的设定
    # netG = MyGenerator(opt.upscale_factor)
    gname = "MyG_3"
    print('# generator parameters:', sum(param.numel() for param in diffu_model.parameters()))
    netD = MyDiscriminator()
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))
    
    criterion = nn.MSELoss(reduction="sum") # (reduction="sum")
    criterion_mse = nn.MSELoss()
    criterion_e = nn.L1Loss(reduction="sum") # (reduction="sum")
    

    # netG.cuda()
    # 把diffusion model的信息加上
    netD.cuda()
    criterion.cuda()
    criterion_mse.cuda()
    criterion_e.cuda()

    print("===> Setting Optimizer")
    optimizerG = optim.Adam(diffu_model.parameters(),lr=opt.lrG)
    optimizerD = optim.Adam(netD.parameters(),lr=opt.lrD)

    schedulerG = optim.lr_scheduler.MultiStepLR(optimizerG, milestones=[1000, 2000, 3000], gamma=0.5)
    
    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading checkpoint '{}'".format(opt.pretrained))
            checkpoint = torch.load(opt.pretrained)
            opt.start_epoch = checkpoint["epoch"]
            best_vec_loss = checkpoint["best_vec_loss"]
            best_img_loss = checkpoint["best_img_loss"]
            best_loss = checkpoint["best_loss"]
            diffu_model.load_state_dict(checkpoint["difffusion"])
            netD.load_state_dict(checkpoint["netD"])
            optimizerG.load_state_dict(checkpoint["optimizerDiffu"])
            optimizerD.load_state_dict(checkpoint["optimizerD"])
        else:
            print("=> no model found at '{}'".format(opt.pretrained))
    else:
        best_loss = float('inf')
        best_vec_loss = float('inf')
        best_img_loss = float('inf')

    results = {'d_loss':[], 'image_loss': [], 'g_loss': [], 'f_loss': [], 'ssim': [], 'vec_mse':[],'img_mse':[]}
    
    
    for epoch in range(opt.start_epoch + 1, opt.num_epochs + opt.start_epoch + 1):
        train_bar = tqdm(train_loader)

        diffu_model.train()
        netD.train()

        for lr, hr in train_bar:
            lr, hr = Variable(lr, requires_grad=True), Variable(hr, requires_grad=False)
            lr = lr.cuda()
            hr = hr.cuda()
            
            #             ############################
            #             # 保存 target 的图片
            #             ############################
            #             for i in range(hr.shape[0]):
            #                 z = hr[i, :, :, :]
            #                 transform = T.ToPILImage()
            #                 z = transform(z)
            #                 if (epoch == 1) and (i % 3 == 0):
            #                     z.save("./scp/trg/trg_{}_{}.png".format(epoch, i))
            #                 elif (epoch % 100 == 0) and (i % 3 == 0):
            #                     z.save("./scp/trg/trg_{}_{}.png".format(epoch, i))

            #             ############################
            #             # 保存 lr 的图片
            #             ############################
            #             for i in range(lr.shape[0]):
            #                 k = lr[i, :, :, :]
            #                 transform = T.ToPILImage()
            #                 k = transform(k)
            #                 if (epoch == 1) and (i % 3 == 0):
            #                     k.save("./scp/lr/lr_{}_{}.png".format(epoch, i))
            #                 elif (epoch % 100 == 0) and (i % 3 == 0):
            #                     k.save("./scp/lr/lr_{}_{}.png".format(epoch, i))
                    
            # 这里把diffusion加上
            t = netG.sample_timesteps(hr.shape[0]).cuda()
            # t = torch.tensor(np.repeat(t, hr.shape[0])).cuda()
            x_t, noise = netG.noise_images(hr, t)
            
            # 保存 noise 的图片
            # for i in range(x_t.shape[0]):
            #     k = x_t[i, :, :, :]
            #     transform = T.ToPILImage()
            #     k = transform(k)
            #     k.save("./scp/noise/noise_{}_{}.png".format(epoch, i))
            predicted_noise = diffu_model(x = x_t, lr = lr, t = t)
            f_loss = criterion(predicted_noise, noise)
            # 学习到了predicted noise
            # 那我们把原有的x0减去pred noise，再upscale 得到高清图片
            # 这里把sr_img的生成改了
            sr_img = sr_images(xt = x_t, lr = lr, model = diffu_model, epoch = epoch, mode = 'train')
            # print('sr_img', sr_img)
            # print('sr_img', sr_img.shape)
            
            
            ############################
            # (1) Update embedding block:
            ###########################
            netD.zero_grad()
            real_out, fake_out = netD(hr, sr_img)
            # default: fake_out = netD(sr_img.detach())
            # print('sr', sr_img.shape)
            #fake_out = netD(sr_img)
            d_loss = criterion_e(real_out, fake_out)
            #print('d_loss', d_loss)
            #print('real_out', real_out)
            #print('fake_out', fake_out)
            # print('============')
            d_loss.backward()
            optimizerD.step()

            ############################
            # (2) Update SR network
            ###########################
            optimizerG.zero_grad()
            image_loss = criterion(sr_img, hr)

            alpha = 0.1
            # g_loss = 0.1*image_loss+d_loss.data*alpha
            g_loss = (image_loss + f_loss*alpha) #  + d_loss.data*alpha 
            g_loss.backward()
            optimizerG.step()
            schedulerG.step()
            
            train_bar.set_description(desc='[%d/%d] d_loss: %.3f g_loss: %.2f f_loss: %.1f image_loss: %.1f' % (
                epoch, opt.num_epochs + opt.start_epoch, d_loss.item(), g_loss.item(), f_loss.item(), image_loss.item()))

        diffu_model.eval()
        netD.eval()
        out_path = 'training_results/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        with torch.no_grad():
            valing_results = {'vec_mse': 0, 'img_mse': 0, 'ssim': 0}
            val_bar = tqdm(val_loader)
            for j, batch in enumerate(val_bar):
                val_lr, val_hr = Variable(batch[0]), Variable(batch[1], requires_grad=False)
                val_lr = val_lr.cuda()
                val_hr = val_hr.cuda()
                
                val_t = netG.sample_timesteps(val_hr.shape[0]).cuda()
                val_x_t, val_noise = netG.noise_images(val_hr, val_t)
                val_predicted_noise = diffu_model(x = val_x_t,lr = val_lr, t = val_t)
                val_sr = sr_images(xt = val_x_t,lr = val_lr, model = diffu_model, epoch = epoch, mode = 'val')
                
                real_out, fake_out = netD(val_hr, val_sr)
                # fake_out = netD(val_sr)
                vec_mse = criterion_e(real_out, fake_out)
                img_mse = criterion(val_hr, val_sr)

                valing_results['vec_mse'] += vec_mse.item()
                valing_results['img_mse'] += img_mse.item()
                val_sr = val_sr.to(torch.device("cpu"))
                val_hr = val_hr.to(torch.device("cpu"))
                # val_sr = val_sr.cuda()
                # val_hr = val_hr.cuda()

                batch_ssim = pytorch_ssim.ssim(val_sr, val_hr)
                valing_results['ssim'] += batch_ssim.item()
                train_bar.set_description(desc="validating……")

        # save loss\scores\psnr\ssim\
        results['d_loss'].append(d_loss.item())
        results['image_loss'].append(image_loss.item())
        results['g_loss'].append(g_loss.item())

        results['ssim'].append(valing_results['ssim']/len(val_loader))
        results['vec_mse'].append(valing_results['vec_mse']/len(val_loader))
        results['img_mse'].append(valing_results['img_mse']/len(val_loader))

        if epoch % opt.print_freq == 0:
            print("epoch: {}\tvector_loss:{} img_loss:{}".format(epoch, valing_results['vec_mse'] / len(val_loader),
                                                                 valing_results['img_mse'] / len(val_loader)))
        if epoch % opt.save_freq == 0:# and epoch != 0:
            vec_loss = valing_results['vec_mse']/len(val_loader)
            img_loss = (valing_results['img_mse']/len(val_loader))/5
            
            # if vec_loss < best_vec_loss:
            if vec_loss + img_loss < best_loss:
                best_vec_loss = vec_loss
                best_img_loss = img_loss
                best_loss = best_vec_loss + best_img_loss
                is_best = True
            else:
                is_best = False
            # print("Saving the model at iteration {} validation loss {}" \
            #       .format(epoch, vec_loss))
            save_checkpoint({
                "epoch": epoch,
                "best_vec_loss": best_vec_loss,
                "best_img_loss": best_img_loss,
                "best_loss": best_loss,
                "difffusion": diffu_model.state_dict(),
                "netD": netD.state_dict(),
                "optimizerDiffu": optimizerG.state_dict(),
                "optimizerD": optimizerD.state_dict()
            }, is_best, gname)

    with open('statistic/train_result_%s.csv' % gname,'a')as f:
        print(results)
        f_csv = csv.DictWriter(f, results.keys())
        f_csv.writeheader()
        f_csv.writerow(results)

def save_checkpoint(state, is_best, name):
    filename = "checkpoint/dma_checkpoint_%s_epoch_%d.pt" % (name,state["epoch"])
    # torch.save(state, filename)
    if is_best:
        print("saving the epoch {} as best model".format(state["epoch"]))
        torch.save(state, filename)
        shutil.copyfile(filename, 'checkpoint/dma_bestmodel_%s.pt'%name)
    # print("Checkpoint saved to {}".format(filename))

if __name__ == "__main__":
    main()
