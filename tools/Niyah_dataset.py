import numpy as np
import torch
import os
import random
import glob
from PIL import Image
from torch.utils.data import Dataset
import cv2


class DatasetWithGT(Dataset):
    def __init__(self, datas_dir, masks_dir, transform=None, in_size=512):
        super(DatasetWithGT, self).__init__()
        self.data_dir = datas_dir
        self.mask_dir = masks_dir
       
        self.transform = transform
        self.label_path_list = glob.glob(os.path.join(self.mask_dir, "*.*"))
        self.img_path_list = glob.glob(os.path.join(self.data_dir, "*.*"))
        self.in_size = in_size

    def __getitem__(self, index):
        path_label = self.label_path_list[index]
        path_img = self.img_path_list[index]
        img_pil = Image.open(path_img)
        label_pil = Image.open(path_label)
        image = self.transform(img_pil)
        mask = self.transform(label_pil)

        return image, mask

    def __len__(self):
        return len(self.label_path_list)


class DatasetWithoutGT(Dataset):
    def __init__(self, datas_dir, transform=None, in_size=512):
        super(DatasetWithoutGT, self).__init__()
        self.data_dir = datas_dir
        self.transform = transform
        self.img_path_list = glob.glob(os.path.join(self.data_dir, "*.*"))
        self.in_size = in_size

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        img_pil = Image.open(img_path).convert('RGB')

        img_chw_tensor = self.transform(img_pil)

        return img_chw_tensor

    def __len__(self):
        return len(self.img_path_list)


class DADatasetWithoutGT(Dataset):
    def __init__(self, source_datas_dir, source_masks_dir, tar_datas_dir, transform=None, in_size=512):
        super(DADatasetWithoutGT, self).__init__()
        self.source_data_dir = source_datas_dir
        self.source_mask_dir = source_masks_dir
        self.tar_data_dir =tar_datas_dir
        self.transform = transform
        self.source_path_list = glob.glob(os.path.join(self.source_data_dir, "*.*"))
        self.source_mask_list = glob.glob(os.path.join(self.source_mask_dir, "*.*"))
        self.tar_path_list = glob.glob(os.path.join(self.tar_data_dir, "*.*"))
        self.in_size = in_size

    def __getitem__(self, index):
        source_img_path = self.source_path_list[index]
        source_img_pil = Image.open(source_img_path).convert('RGB')
        source_mask_path = self.source_mask_list[index]
        source_mask_pil = Image.open(source_mask_path).convert('RGB')

        tar_img_path = self.tar_path_list[index]
        tar_img_pil = Image.open(tar_img_path).convert('RGB')

        source_img= self.transform(source_img_pil)
        source_mask = self.transform(source_mask_pil)
        tar_img = self.transform(tar_img_pil)

        return source_img, source_mask, tar_img

    def __len__(self):
        return min(len(self.source_path_list), len(self.tar_path_list))

class DADatasetWithGT(Dataset):
    def __init__(self, source_datas_dir, source_masks_dir, tar_datas_dir, tar_masks_dir, transform=None, in_size=512):
        super(DADatasetWithGT, self).__init__()
        self.source_data_dir = source_datas_dir
        self.source_mask_dir = source_masks_dir
        self.tar_data_dir =tar_datas_dir
        self.tar_mask_dir = tar_masks_dir
        self.transform = transform
        self.source_path_list = glob.glob(os.path.join(self.source_data_dir, "*.*"))
        self.source_mask_list = glob.glob(os.path.join(self.source_mask_dir, "*.*"))
        self.tar_path_list = glob.glob(os.path.join(self.tar_data_dir, "*.*"))
        self.tar_mask_list = glob.glob(os.path.join(self.tar_mask_dir, "*.*"))
        self.in_size = in_size

    def __getitem__(self, index):
        source_img_path = self.source_path_list[index]
        source_img_pil = Image.open(source_img_path).convert('RGB')
        source_mask_path = self.source_mask_list[index]
        source_mask_pil = Image.open(source_mask_path).convert('RGB')

        tar_img_path = self.tar_path_list[index]
        tar_img_pil = Image.open(tar_img_path).convert('RGB')
        tar_mask_path = self.tar_path_list[index]
        tar_mask_pil = Image.open(tar_mask_path).convert('RGB')

        source_img= self.transform(source_img_pil)
        source_mask = self.transform(source_mask_pil)
        tar_img = self.transform(tar_img_pil)
        tar_mask=self.transform(tar_mask_pil)

        return source_img, source_mask, tar_img, tar_mask

    def __len__(self):
        return min(len(self.source_path_list), len(self.tar_path_list))

