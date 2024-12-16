# -*- coding: utf-8 -*-

import os
import cv2
import time
import glob
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tools.Niyah_dataset import DADatasetWithGT
import torch.optim as optim
from tools.common_tools import set_seed
from UNet_Models import *
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
# from torchvision.transforms import Resize, Grayscale
import argparse
from PIL import Image
# import math
import itertools
# import datetime
import time
from torch.autograd import Variable
from EdgeAtt_CycleGAN import Generator, Discriminator
from CycleGAN_util import LambdaLR, ReplayBuffer, weights_init_normal
from skimage import measure
from skimage import morphology


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_seed()  # 设置随机种子

transform_ = transforms.Compose([transforms.Grayscale(),
                                 transforms.Resize((512, 512)),
                                 transforms.ToTensor(),
                                ])

def compute_dice(y_pred, y_true):
    """
    :param y_pred: 4-d tensor, value = [0,1]
    :param y_true: 4-d tensor, value = [0,1]
    :return:
    """
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    y_pred, y_true = np.round(y_pred).astype(int), np.round(y_true).astype(int)
    return np.sum(y_pred[y_true == 1]) * 2.0 / (np.sum(y_pred) + np.sum(y_true))

def tu_seg(masks):
    re_mask = []
    for m in range(len(masks)):
        th1 = masks[m]
        res1 = np.zeros((th1.shape[0], th1.shape[1]))
        res2 = np.zeros((th1.shape[0], th1.shape[1]))
        label_img, num = measure.label(th1[:, :], background=0, return_num=True, connectivity=2)
        props = measure.regionprops(label_img)
        areas = np.zeros(num)
        for i in range(0, num):
            areas[i] = props[i].area
        bb = areas.tolist()
        # print('联通区域的面积分别是：', bb);
        c = []
        a = bb.copy()
        # 寻找最大联通区域的的label值；
        c.append(bb.index(max(bb)))
        # 寻找第二大联通区域的的label值；
        a.remove(max(bb))
        c.append(bb.index(max(a)))
        # print(c)
        res1[np.where(label_img == c[0] + 1)] = 255
        res2[np.where(label_img == c[1] + 1)] = 255
        for j in range(0, res1.shape[0]):
            if len([np.where(res1[j, :] == 255)][0][0]) > 0:
                y0 = j
                break
        for k in range(0, res2.shape[0]):
            if len([np.where(res2[k, :] == 255)][0][0]) > 0:
                y1 = k
                break
        if min([np.where(res1[y0, :] == 255)][0][0]) > min([np.where(res2[y1, :] == 255)][0][0]):
            right_lung=res2
            left_lung=res1
        else:
            right_lung=res1
            left_lung=res2
        mask_m = right_lung + left_lung
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask_m = cv2.morphologyEx(mask_m, cv2.MORPH_CLOSE, kernel)  ## 有缺陷，填补缺陷
        mask_m = transform_(Image.fromarray(mask_m))
        mask_m = torch.unsqueeze(mask_m, 0)
        re_mask.append(mask_m)
    return torch.cat(re_mask, dim=0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=80, help="epoch from which to start lr decay")
    parser.add_argument("--img_height", type=int, default=512, help="size of image height")
    parser.add_argument("--img_width", type=int, default=512, help="size of image width")
    parser.add_argument("--input_nc", type=int, default=1, help="number of input image channels")
    parser.add_argument("--output_nc", type=int, default=1, help="number of output image channels")
    parser.add_argument("--checkpoint_interval", type=int, default=3, help="interval between saving model checkpoints")
    parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
    parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
    parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
    parser.add_argument("--mask_thres", type=float, default=0.5, help="mask_thres")
    opt = parser.parse_args()
    # opt = parser.parse_args(args=[])                 ## 在colab中运行时，换为此行
    print(opt)
  
    tar_img_dir = '/home/niuyi/dataset/UDA/ShenZhen/image'
    tar_mask_dir = '/home/niuyi/dataset/UDA/ShenZhen/mask'
    source_img_dir = '/home/niuyi/dataset/UDA/Meg/image'
    source_mask_dir = '/home/niuyi/dataset/UDA/Meg/mask'
    TIME = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    pred_save_path = './result/EdgeAtt_ShenZhen_result/{0}'.format(TIME)
    model_save_path = './model/EdgeAtt_ShenZhen_model/{0}'.format(TIME)

    if not os.path.exists(pred_save_path):
        os.makedirs(pred_save_path)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    if not os.path.exists('./logs/EdgeAtt_ShenZhen_logs/{0}'.format(TIME)):
        os.makedirs('./logs/EdgeAtt_ShenZhen_logs/{0}'.format(TIME))

    # step 1
    train_set = DADatasetWithGT(source_img_dir, source_mask_dir, tar_img_dir, tar_mask_dir, transform=transform_)
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, drop_last=True)
    
    # 2. model
    unet = AttU_Net(img_ch=1, output_ch=1)
    unet_model_path = "./model/Meg_Aug_AttUnet_checkpoint.pkl"
    unet.load_state_dict(torch.load(unet_model_path, map_location="cpu")['model_state_dict'])
    unet.to(device)

    ## 创建生成器，判别器对象
    G_ST = Generator(opt.input_nc, opt.output_nc, opt.n_residual_blocks).to(device)
    G_TS = Generator(opt.input_nc, opt.output_nc, opt.n_residual_blocks).to(device)
    D_T = Discriminator(opt.input_nc).to(device)
    D_S = Discriminator(opt.input_nc).to(device)
    D_mask = Discriminator(opt.input_nc).to(device)

    ## 损失函数
    ## MES 二分类的交叉熵
    ## L1loss 相比于L2 Loss保边缘
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    # criterion_identity = torch.nn.L1Loss()
    criterion_mask = torch.nn.MSELoss()
    criterion_edge = torch.nn.BCELoss()

    G_ST.apply(weights_init_normal)
    G_TS.apply(weights_init_normal)
    D_T.apply(weights_init_normal)
    D_S.apply(weights_init_normal)
    D_mask.apply(weights_init_normal)

    ## 定义优化函数,优化函数的学习率为0.0002
    optimizer_G = torch.optim.Adam(itertools.chain(G_ST.parameters(), G_TS.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D_T = torch.optim.Adam(D_T.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D_S = torch.optim.Adam(D_S.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D_mask = torch.optim.Adam(D_mask.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    ## 学习率更行进程
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D_T = torch.optim.lr_scheduler.LambdaLR(optimizer_D_T, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D_S = torch.optim.lr_scheduler.LambdaLR(optimizer_D_S, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)  
    lr_scheduler_D_mask = torch.optim.lr_scheduler.LambdaLR(optimizer_D_mask, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

    ## 先前生成的样本的缓冲区
    fake_T_buffer = ReplayBuffer()  
    fake_S_buffer = ReplayBuffer()
    
    for epoch in range(opt.n_epochs):
        
        train_G_loss_total =0.
        train_cycle_loss_total =0.
        train_edge_loss_total =0.
        # train_identity_loss_total =0.
        train_GAN_loss_total =0.
        train_mask_loss_total =0.
        train_DT_loss_total =0.
        train_DS_loss_total =0.
        train_DMask_loss_total =0.
        train_dice_total =0.
        n_batch=0

        for iter, (s_img, s_mask, t_img, t_mask) in enumerate(train_loader):
            G_TS.train()
            G_ST.train()
            D_T.train()
            D_S.train()
            D_mask.train()
            unet.eval()

            real_s, s_mask, real_t, t_mask = s_img.cuda(), s_mask.cuda(), t_img.cuda(), t_mask.cuda()
            ## 全真，全假的标签
            valid = Variable(torch.ones(real_s.size(0), dtype=torch.float32), requires_grad=False).cuda()     
            fake = Variable(torch.zeros(real_s.size(0), dtype=torch.float32), requires_grad=False).cuda() 

            # ## Identity loss 
            # real_s_hat, _ = G_ST(real_s)
            # real_t_hat, _ = G_TS(real_t)                                            
            # loss_id_T = criterion_identity(real_t_hat, real_t)         
            # loss_id_S = criterion_identity(real_s_hat, real_s)
            # loss_identity = (loss_id_S + loss_id_T) / 2                   ## Identity loss 

            ## GAN loss
            fake_S, pre_edge_T = G_TS(real_t)                                      
            loss_GAN_TS = criterion_GAN(D_S(fake_S), valid)            
            fake_T, pre_edge_S = G_ST(real_s)                                         
            loss_GAN_ST = criterion_GAN(D_T(fake_T), valid)               
            loss_GAN = (loss_GAN_TS + loss_GAN_ST) / 2                    ## GAN loss

            mask_T = unet(fake_S).ge(opt.mask_thres)
        
            if epoch>=70:
                try:
                    mask_T = tu_seg(torch.squeeze(mask_T).detach().cpu().numpy()).cuda()
                except:
                    pass
                mask_T_edge = mask_T
            else:
                mask_T_edge = unet(real_t.detach()).ge(opt.mask_thres)
                try:
                    mask_T_edge = tu_seg(torch.squeeze(mask_T_edge).detach().cpu().numpy()).cuda()
                except:
                    pass        
            mask_T_edge = torch.tensor(mask_T_edge, dtype=torch.float32)
            
            
            t_edge = torch.abs(torch.roll(mask_T_edge, 3, 3)-mask_T_edge) + torch.abs(torch.roll(mask_T_edge, 3, 2)-mask_T_edge)
            s_edge = torch.abs(torch.roll(s_mask, 3, 3)-s_mask) + torch.abs(torch.roll(s_mask, 3, 2)-s_mask)
            loss_edge = criterion_edge(pre_edge_T, t_edge)+criterion_edge(pre_edge_S, s_edge)        ## Edge loss

            # mask损失
            mask_T = torch.tensor(mask_T, dtype=torch.float32)
            loss_mask = criterion_mask(D_mask(mask_T), valid)

            # Cycle loss 循环一致性损失                                                 
            recov_T, _ = G_ST(fake_S)                                        
            loss_cycle_T = criterion_cycle(recov_T, real_t)               
            recov_S, _ = G_TS(fake_T)
            loss_cycle_S = criterion_cycle(recov_S, real_s)
            loss_cycle = (loss_cycle_S + loss_cycle_T) / 2         

            # Total loss                                                  ## 就是上面所有的损失都加起来
            loss_G = loss_GAN + opt.lambda_cyc * loss_cycle + loss_mask + 6*loss_edge
            # loss_G = loss_GAN + opt.lambda_cyc * loss_cycle + opt.lambda_id * loss_identity + loss_mask + 10*loss_edge
            optimizer_G.zero_grad()                                       ## 在反向传播之前，先将梯度归0
            loss_G.backward()                                             ## 将误差反向传播
            optimizer_G.step()                                            ## 更新参数


            ## -----------------------
            ## Train Discriminator T
            ## 分为两部分：1、真的图像判别为真；2、假的图像判别为假
            loss_real = criterion_GAN(D_T(real_t), valid)
            fake_T_ = fake_T_buffer.push_and_pop(fake_T)
            loss_fake = criterion_GAN(D_T(fake_T_.detach()), fake)
            # Total loss
            loss_D_T = (loss_real + loss_fake) / 2
            optimizer_D_T.zero_grad()                                     ## 在反向传播之前，先将梯度归0
            loss_D_T.backward()                                           ## 将误差反向传播
            optimizer_D_T.step()                                          ## 更新参数

            ## -----------------------
            ## Train Discriminator S
            loss_real = criterion_GAN(D_S(real_s), valid)
            fake_S_ = fake_S_buffer.push_and_pop(fake_S)
            loss_fake = criterion_GAN(D_S(fake_S_.detach()), fake)
            # Total loss
            loss_D_S = (loss_real + loss_fake) / 2
            optimizer_D_S.zero_grad()                                     ## 在反向传播之前，先将梯度归0
            loss_D_S.backward()                                           ## 将误差反向传播
            optimizer_D_S.step()                                          ## 更新参数

             ## -----------------------
            ## Train Discriminator Mask
            loss_S= criterion_mask(D_mask(s_mask), valid)
            # fake_T_ = fake_T_buffer.push_and_pop(fake_T)
            # mask_T = torch.tensor(unet(fake_T_.detach()).ge(opt.mask_thres), dtype = torch.float32)
            loss_T = criterion_GAN(D_mask(mask_T.detach()), fake)
            # Total loss
            loss_D_mask = (loss_T + loss_S) / 2
            optimizer_D_mask.zero_grad()                                     ## 在反向传播之前，先将梯度归0
            loss_D_mask.backward()                                           ## 将误差反向传播
            optimizer_D_mask.step()                                          ## 更新参数

            train_G_loss_total += loss_G.detach().cpu().numpy()
            train_cycle_loss_total += loss_cycle.detach().cpu().numpy()
            train_edge_loss_total += loss_edge.detach().cpu().numpy()
            # train_identity_loss_total += loss_identity.detach().cpu().numpy()
            train_GAN_loss_total += loss_GAN.detach().cpu().numpy()
            train_mask_loss_total += loss_mask.detach().cpu().numpy()
            train_DT_loss_total += loss_D_T.detach().cpu().numpy()
            train_DS_loss_total += loss_D_S.detach().cpu().numpy()
            train_DMask_loss_total += loss_D_mask.detach().cpu().numpy()
            train_dice = compute_dice(mask_T.detach().cpu().numpy(), t_mask.detach().cpu().numpy())
            train_dice_total += train_dice
            n_batch +=1

            
            # if iter == len(train_loader)-1 and (epoch + 1) % opt.checkpoint_interval == 0:
            vutils.save_image(torch.cat([real_t.cpu(), fake_S.cpu(), mask_T.cpu(), t_edge.cpu(), pre_edge_T.ge(opt.mask_thres).cpu()], dim=0), './result/EdgeAtt_ShenZhen_result/{0}/epoch_{1}.png'.format(TIME, epoch))
                # G_TS.eval()
                # with torch.no_grad():
                #     tmp_fake_s, tmp_pre_edge_t = G_TS(real_t)
                    # vutils.save_image(torch.cat([real_t, tmp_fake_s, unet(tmp_fake_s).ge(opt.mask_thres), t_edge, tmp_pre_edge_t], dim=0), './result/EdgeAtt_ShenZhen_result/{0}/epoch_{1}.png'.format(TIME, epoch))

            print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] cycle_loss: {:.4f} edge_loss: {:.4f}  " 
                    "GAN_loss: {:.4f} mask_loss: {:.4f}  G_loss:{:.4f} D_T_loss:{:.4f} D_S_loss:{:.4f} D_mask_loss:{:.4f} train_dice:{:.4f}".format(epoch, opt.n_epochs, iter + 1, len(train_loader),
                    loss_cycle.detach().cpu().numpy(), loss_edge.detach().cpu().numpy(), loss_GAN.detach().cpu().numpy(), loss_mask.detach().cpu().numpy(),
                    loss_G.detach().cpu().numpy(), loss_D_T.detach().cpu().numpy(), loss_D_S.detach().cpu().numpy(), loss_D_mask.detach().cpu().numpy(), train_dice))

            torch.cuda.empty_cache()

        lr_scheduler_G.step()
        lr_scheduler_D_T.step()
        lr_scheduler_D_S.step()
        lr_scheduler_D_mask.step()


        with SummaryWriter('./logs/EdgeAtt_ShenZhen_logs/{0}'.format(TIME)) as writer:
            writer.add_scalar('train/G_loss', train_G_loss_total/n_batch, epoch)
            writer.add_scalar('train/cycle_loss', train_cycle_loss_total/n_batch, epoch)
            writer.add_scalar('train/edge_loss', train_cycle_loss_total/n_batch, epoch)
            # writer.add_scalar('train/identity_loss', train_identity_loss_total/n_batch, epoch)
            writer.add_scalar('train/GAN_loss', train_GAN_loss_total/n_batch, epoch)
            writer.add_scalar('train/mask_loss', train_mask_loss_total/n_batch, epoch)
            writer.add_scalar('train/D_T_loss', train_DT_loss_total/n_batch, epoch)
            writer.add_scalar('train/D_S_loss', train_DS_loss_total/n_batch, epoch)
            writer.add_scalar('train/D_mask_loss', train_DMask_loss_total/n_batch, epoch)
            writer.add_scalar('train/dice', train_dice_total/n_batch, epoch)
            writer.close()

        if (epoch + 1) % opt.checkpoint_interval == 0 and epoch>=50:
            torch.save(G_TS.state_dict(), os.path.join(model_save_path, "G_TS_{}_epoch.pkl".format(epoch)))
            torch.save(G_ST.state_dict(), os.path.join(model_save_path, "G_ST_{}_epoch.pkl".format(epoch)))
            