import Augmentor
import cv2
import glob
import os
from tqdm import tqdm
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia
from albumentations import Rotate, HorizontalFlip, ShiftScaleRotate, RandomCrop

def load_data(path):
    images = sorted(glob.glob(os.path.join(path, "image/*.*")))
    masks = sorted(glob.glob(os.path.join(path, "lung/*.*")))
    return images, masks

def augment_data(images, masks, save_path, augment=True):
    H = 512
    W = 512
    for image, mask in tqdm(zip(images, masks), total=len(images)):
        image_name = os.path.basename(image).split('.')[0]
        image_extn = os.path.basename(image).split('.')[1]


        image = cv2.imread(image, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask, cv2.IMREAD_COLOR)
        if augment==True:
            aug1 = Rotate(20, border_mode=0,value=0,mask_value=0,p=1)
            augmented1 = aug1(image=image, mask=mask)
            image1 = augmented1['image']
            mask1 = augmented1['mask']

            aug2 = HorizontalFlip(p=1)
            augmented2 = aug2(image=image, mask=mask)
            image2 = augmented2['image']
            mask2 = augmented2['mask']

            aug3 = ShiftScaleRotate(shift_limit=0.2, rotate_limit=0, scale_limit=0, border_mode=0,value=0,mask_value=0,p=1)
            augmented3 = aug3(image=image, mask=mask)
            image3 = augmented3['image']
            mask3 = augmented3['mask']

            aug4 = ShiftScaleRotate(scale_limit=0.25, shift_limit=0, rotate_limit=0,border_mode=0,value=0,mask_value=0, p=1)
            augmented4 = aug4(image=image, mask=mask)
            image4 = augmented4['image']
            mask4 = augmented4['mask']

            aug5 = RandomCrop(height=image.shape[0]-250, width=image.shape[1]-250, p=1.0)
            augmented5 = aug5(image=image, mask=mask)
            image5 = augmented5['image']
            mask5 = augmented5['mask']

            save_images = [image, image1, image2, image3, image4, image5]
            save_masks = [mask, mask1, mask2, mask3, mask4, mask5]
        else:
            save_images = [image]
            save_masks = [mask]

        idx = 0
        for i, m in zip(save_images, save_masks):
            tmp_img_name = f"{image_name}_{idx}.{image_extn}"
            tmp_mask_name = f"{image_name}_{idx}.{image_extn}"

            image_path = os.path.join(save_path, "Aug_images", tmp_img_name)
            mask_path = os.path.join(save_path, "Aug_masks", tmp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)
            idx += 1



if __name__=="__main__":
    path = "/home/niuyi/dataset/lung_segmnetation/MC/train"
    images, masks = load_data(path)
    # images=["/home/niuyi/dataset/lung_segmnetation/MC/train/img/MCUCXR_0141_1.png"]
    # masks=["/home/niuyi/dataset/lung_segmnetation/MC/train/GT/MCUCXR_0141_1.png"]


    image_save_dir = os.path.join(path, "Aug_images")
    mask_save_dir = os.path.join(path, "Aug_masks")
    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir)
    if not os.path.exists(mask_save_dir):
        os.makedirs(mask_save_dir)

    augment_data(images, masks, path, augment=True)



