# -*- coding: utf-8 -*-
import os
import time
import random
import torch.nn as nn
import torch
import glob
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from tools.Niyah_dataset import DatasetWithGT, DatasetWithoutGT
from matplotlib import pyplot as plt
import torch.optim as optim
import torchvision.models as models
from tools.common_tools import set_seed
from tools.my_dataset import PortraitDataset
# from tools.unet import UNet
from UNet_Models import *
# from CycleGAN_Models import Generator
from EdgeAtt_CycleGAN import Generator
from torch.utils.data import Dataset
import shutil
import cv2
from skimage import measure
from skimage import morphology
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

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

def compute_jaccard(y_pred, y_true):
    """
    :param y_pred: 4-d tensor, value = [0,1]
    :param y_true: 4-d tensor, value = [0,1]
    :return:
    """
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    y_pred, y_true = np.round(y_pred).astype(int), np.round(y_true).astype(int)
    return np.sum(y_pred[y_true == 1]) / (np.sum(y_pred) + np.sum(y_true)-np.sum(y_pred[y_true == 1]))

def get_img_name(img_dir, format="jpg"):
    """
    获取文件夹下format格式的文件名
    :param img_dir: str
    :param format: str
    :return: list
    """
    file_names = os.listdir(img_dir)
    img_names = list(filter(lambda x: x.endswith(format), file_names))
    img_names = list(filter(lambda x: not x.endswith("matte.png"), img_names))

    if len(img_names) < 1:
        raise ValueError("{}下找不到{}格式数据".format(img_dir, format))
    return img_names

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
    # save_mask_dir = '/home/niuyi/dataset/original_dataset/CXR_ipmi/normal_DA_mask'
    # save_mask_dir = './result/EdgeAtt_IPMI_result/pred_mask'
    # save_fakeImg_dir = './result/EdgeAtt_IPMI_result/fake_image'

    # if not os.path.exists(save_mask_dir):
    #     os.makedirs(save_mask_dir)
    # if not os.path.exists(save_fakeImg_dir):
    #     os.makedirs(save_fakeImg_dir)

    time_total = 0
    mask_thres = .5

    # 1. data
    transform_ = transforms.Compose([transforms.Grayscale(),
                                     transforms.Resize((512, 512)),
                                     transforms.ToTensor(),
                                     ])

    # ShenZhen
    img_dir = '/home/niuyi/dataset/UDA/ShenZhen'
    datas_dir = os.path.join(img_dir, 'image')
    masks_dir = os.path.join(img_dir, 'mask')
    test_set = DatasetWithGT(datas_dir, masks_dir, transform=transform_)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, drop_last=False)
    img_names = os.listdir(datas_dir)

    #IPMI
    # # img_dir = '/home/niuyi/dataset/original_dataset/CXR_ipmi/normal'
    # img_dir = '/home/niuyi/dataset/original_dataset/CXR_ipmi/abnormal'
    # test_set = DatasetWithoutGT(img_dir, transform=transform_)
    # test_loader = DataLoader(test_set, batch_size=1, shuffle=False, drop_last=False)
    # img_names = os.listdir(img_dir)


    # 2. model
    model_path = "./model/Meg_Aug_AttUnet_checkpoint.pkl"
    unet = AttU_Net(img_ch=1, output_ch=1)
    unet.load_state_dict(torch.load(model_path, map_location="cpu")['model_state_dict'])
    unet.to(device)
    unet.eval()

    G_TS = Generator(1,1,9)
    G_TS.load_state_dict(torch.load("./model/EdgeAtt_ShenZhen_model/G_TS_checkpoint.pkl"))
    # G_TS.load_state_dict(torch.load("./model/EdgeAtt_IPMI_model/G_TS_checkpoint.pkl"))
    G_TS.to(device)
    G_TS.eval()

    total_dice = 0.
    total_jaccard = 0.
    n_samples=0
    #ShenZhen
    for i, (image, GT_mask) in enumerate(test_loader):
        if torch.cuda.is_available():
            image, GT_mask = image.to(device), GT_mask.to(device)
        image, _ = G_TS(image)
        output = unet(image)
        pred = output.ge(mask_thres)
        try:
            pred_mask = tu_seg(torch.squeeze(pred, 1).detach().cpu().numpy())
        except:
            pred_mask = pred
        dice = compute_dice(pred_mask.detach().cpu().numpy(), GT_mask.detach().cpu().numpy())
        jaccard = compute_jaccard(pred_mask.detach().cpu().numpy(), GT_mask.detach().cpu().numpy())
        total_dice += dice
        total_jaccard += jaccard
        n_samples +=1
        print(i)
        # vutils.save_image(pred_mask.float(), save_mask_dir + '/{0}_mask.png'.format(img_names[i].split('.')[0]))
        # vutils.save_image(image, save_fakeImg_dir + '/{0}_fake.png'.format(img_names[i].split('.')[0]))
    print("mean_dice:{0}  mean_jaccard:{1}".format(total_dice/n_samples, total_jaccard/n_samples))

    # IPMI
    # for i, image in enumerate(test_loader):
    #     if torch.cuda.is_available():
    #         image = image.to(device)
    #     image, _ = G_TS(image)
    #     output = unet(image)
    #     pred = output.ge(mask_thres)
    #     try:
    #         pred_mask = tu_seg(torch.squeeze(pred, 1).detach().cpu().numpy())
    #     except:
    #         pred_mask =pred
        
    #     print(i)
    #     # vutils.save_image(pred_mask.float(), save_mask_dir + '/{0}_mask.png'.format(img_names[i].split('.')[0]))
    #     vutils.save_image(image, save_fakeImg_dir + '/{0}.png'.format(img_names[i].split('.')[0]))


