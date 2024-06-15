import os
import cv2
import torch
import jpegio
import tempfile
import numpy as np
import pickle
from PIL import Image
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
import albumentations as A
import logging
import random
from glob import glob

train_transform = A.Compose([
    # A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),#水平翻转
    A.OneOf([
        A.VerticalFlip(p=0.5),#垂直翻转
        A.RandomRotate90(p=0.5),#旋转90°
        A.RandomBrightnessContrast(p=0.2, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)),#亮度、对比度
        A.HueSaturationValue(p=0.2, hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2),#色彩
        A.ShiftScaleRotate(p=0.2, shift_limit=0.0625, scale_limit=0.2, rotate_limit=20),#仿射变换：随机对图像进行平移、缩放和旋转。shift_limit 定义了平移的范围（这里是图像宽度和高度的 6.25%），scale_limit 定义了缩放的范围（这里是原始尺寸的 20%）
        A.CoarseDropout(p=0.2),# 粗暴丢弃：随机丢弃图像中的一部分区域，这可以增加模型对遮挡和噪声的鲁棒性。
        A.Transpose(p=0.5)#将图像的行和列互换，即宽度和高度互换。
    ]),
    A.Normalize(mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
val_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
class DocTamperDataset(Dataset):
    '''
    basic dataset
    #包括了课程学习策略
    '''

    def __init__(self, imgs_dir, masks_dir, q_level, mode='train'):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.mode = mode
        self.q_level = q_level # [0,1,2,3,4,5]
        with open('/data/jinrong/wcw/PS/DocTamper/qt_table.pk', 'rb') as fpk:
            pks = pickle.load(fpk)
        self.pks = {}
        for k, v in pks.items():
            self.pks[k] = torch.LongTensor(v)

        self.ids = sorted([os.path.splitext(file)[0] for file in os.listdir(imgs_dir)
                    if not file.startswith('.')])#列出imgs_dir目录下的所有非隐藏文件的基本名称
        self.masks_dir_ids = sorted([os.path.splitext(file)[0] for file in os.listdir(masks_dir)
                             if not file.startswith('.')])  # 列出imgs_dir目录下的所有非隐藏文件的基本名称
        # 确保图片和标签名称顺序一致,不一致则报错
        assert self.ids == self.masks_dir_ids, 'Image file names are not the same as label file names'
        logging.info(f'Creating dataset with {len(self.ids)} examples')
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        # 读取图片
        idx = self.ids[i]
        img_file = glob(self.imgs_dir + idx + '.*')
        mask_file = glob(self.masks_dir + idx + '.*')
        # 随机压缩
        rand_q = random.randint(100-5*self.q_level, 100)#从95-100，衰减为70-100
        use_qtb = self.pks[rand_q]

        with tempfile.NamedTemporaryFile(delete=True) as tmp:
            im = Image.open(img_file).convert("L")
            im.save(tmp, "JPEG",quality=rand_q)
            jpg = jpegio.read(tmp.name)
            dct = jpg.coef_arrays[0].copy()
            im = im.convert('RGB')
            im = np.array(im)
            dct_coef = torch.tensor(np.clip(np.abs(dct), 0, 20))
            if self.mode == 'train':
                transformed = train_transform(image=im, mask=cv2.imread(mask_file))
            elif self.mode == 'val':
                transformed = val_transform(image=im, mask=cv2.imread(mask_file))
        return {
            'image': transformed['image'],
            'label': transformed['mask'].long,
            'dct': np.clip(np.abs(dct_coef),0,20),
            'qtb':use_qtb,
            # 'q':rand_q
        }
