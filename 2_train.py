# -*- coding: utf-8 -*-

import os
import time
import torch.nn as nn
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from tools.common_tools import set_seed
from tools.Niyah_dataset import DatasetWithGT
from UNet_Models import *
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from CycleGAN_util import Logger


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_seed()  # 设置随机种子


def compute_dice(y_pred, y_true):
    """
    :param y_pred: 4-d tensor, value = [0,1]
    :param y_true: 4-d tensor, value = [0,1]
    :return:
    """
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    y_pred, y_true = np.round(y_pred).astype(int), np.round(y_true).astype(int)
    dice = np.sum(y_pred[y_true == 1]) * 2.0 / (np.sum(y_pred) + np.sum(y_true))
    return dice

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)

        smooth = 1
        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)
        intersection = input_flat * target_flat
        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N
        return loss


if __name__ == "__main__":

    # config
    LR = 0.001
    BATCH_SIZE = 8
    max_epoch = 200
    start_epoch = 0
    lr_step = 150
    val_interval = 1
    checkpoint_interval = 1
    vis_num = 10
    mask_thres = 0.5
    BASE_DIR = '/home/niuyi/dataset/lung_segmnetation/MC'
    TIME = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    pred_save_path = './result/{0}_Meg_Aug_AttUnet'.format(TIME)
    model_save_path = './model/{0}_Meg_Aug_AttUnet'.format(TIME)

    if not os.path.exists(pred_save_path):
        os.makedirs(pred_save_path)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    if not os.path.exists('./logs/{0}_Meg_Aug_AttUnet'.format(TIME)):
        os.makedirs('./logs/{0}_Meg_Aug_AttUnet'.format(TIME))

    train_dir = os.path.join(BASE_DIR, "train")
    valid_dir = os.path.join(BASE_DIR, "val")
    train_data_dir = os.path.join(train_dir, "Aug_images")
    train_mask_dir = os.path.join(train_dir, "Aug_masks")
    valid_data_dir = os.path.join(valid_dir, "img")
    valid_mask_dir = os.path.join(valid_dir, "GT")

    valid_image_names = os.listdir(valid_data_dir)

    transform_ = transforms.Compose([transforms.Grayscale(),
                                     transforms.Resize((512, 512)),
                                     transforms.ToTensor(),
                                    ])

    # step 1
    train_set = DatasetWithGT(datas_dir=train_data_dir, masks_dir=train_mask_dir, transform=transform_)
    valid_set = DatasetWithGT(datas_dir=valid_data_dir, masks_dir=valid_mask_dir, transform=transform_)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=True, drop_last=False)
   
    logger = Logger(max_epoch, len(train_loader))


    # step 2  
    # net = U_Net(in_ch=1, out_ch=1)
    net = AttU_Net(img_ch=1, output_ch=1)
    net.to(device)

    # step 3
    loss_fn = nn.MSELoss()
    #loss_fn = DiceLoss()
    #loss_fn = dice_coeff()

    # step 4
    #optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=0.1)

    # step 5

    for epoch in range(start_epoch, max_epoch):
        train_loss_total = 0.
        train_dice_total = 0.
        n_batch = 0
        net.train()
        # print(len(train_loader))
        for iter, (inputs, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                inputs, labels = inputs.to(device), labels.to(device)
            # forward
            outputs = net(inputs)
            # backward
            optimizer.zero_grad()
            loss = loss_fn(outputs, labels)
            #loss = dice_coeff(outputs, labels)
            loss.backward()
            optimizer.step()
            # print
            train_dice = compute_dice(outputs.ge(mask_thres).detach().cpu().numpy(), labels.detach().cpu().numpy())
            train_loss_total += loss.detach().cpu().numpy()
            train_dice_total += train_dice
            n_batch += 1
            # print(n_batch)

            if iter == len(train_loader)-1:
                vutils.save_image(torch.cat([inputs, outputs.ge(mask_thres), labels], dim=0), './result/{0}_Meg_Aug_AttUnet/epoch_{1}.png'.format(TIME, epoch))

            # print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] running_loss: {:.4f}, mean_loss: {:.4f} "
            #       "running_dice: {:.4f} lr:{}".format(epoch, max_epoch, iter + 1, len(train_loader), loss.item(),
            #                         train_loss_total/(iter+1), train_dice, scheduler.get_lr()))
            print("  ")

            logger.log({'running_loss': loss, 'running_dice': train_dice})

        scheduler.step()


        with SummaryWriter('./logs/{0}_Meg_Aug_AttUnet'.format(TIME)) as writer:
            writer.add_scalar('train/loss', train_loss_total/n_batch, epoch)
            writer.add_scalar('train/dice', train_dice_total/n_batch, epoch)
            writer.close()

        if (epoch + 1) % checkpoint_interval == 0 and (epoch+1) >= 25:
            checkpoint = {"model_state_dict": net.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch}
            path_checkpoint = os.path.join(model_save_path, "checkpoint_{}_epoch.pkl".format(epoch))
            torch.save(checkpoint, path_checkpoint)

        # validate the model
        if (epoch+1) % val_interval == 0:            
            valid_loss_total = 0.
            valid_dice_total = 0.
            with torch.no_grad():
                net.eval()
                for j, (inputs, labels) in enumerate(valid_loader):
                    if torch.cuda.is_available():
                        inputs, labels = inputs.to(device), labels.to(device)

                    outputs = net(inputs)
                    loss = loss_fn(outputs, labels)
                    # loss = dice_coeff(outputs, labels)
                    valid_loss_total += loss.item()

                    valid_dice = compute_dice(outputs.ge(mask_thres).detach().cpu().numpy(), labels.detach().cpu().numpy())
                    valid_dice_total += valid_dice

                valid_loss_mean = valid_loss_total/len(valid_loader)
                valid_dice_mean = valid_dice_total/len(valid_loader)

                print("Valid:\t Epoch[{:0>3}/{:0>3}] mean_loss: {:.4f} dice_mean: {:.4f}".format(
                    epoch, max_epoch, valid_loss_mean, valid_dice_mean))

                with SummaryWriter('./logs/{0}_Meg_Aug_AttUnet'.format(TIME)) as writer:
                    writer.add_scalar('valid/loss', valid_loss_mean, epoch)
                    writer.add_scalar('valid/dice', valid_dice_mean, epoch)
                    writer.close()

        torch.cuda.empty_cache()
