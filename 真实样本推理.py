# -*- coding: utf-8 -*-

from models.model_auxiliary import *
import math
import numpy as np
import torchvision
import cv2
import torch
from PIL import Image
import jpegio
import os
import tqdm
import pickle
import tempfile
from torch.autograd import Variable
import zipfile


def get_patch(pic_path,patch_size=512):
    '''
        '''
    im = cv2.imread(pic_path)
    H, W = im.shape[:2]
    patch_list = []
    H_num = math.ceil(H / patch_size)
    W_num = math.ceil(W / patch_size)
    for i in range(H_num):
        for j in range(W_num):
            beginH, endH = i * patch_size, (i + 1) * patch_size
            beginW, endW = j * patch_size, (j + 1) * patch_size
            patch = im[beginH:endH, beginW:endW,:]
            patchH, patchW = patch.shape[0:2]
            patch = np.pad(patch, ((0, patch_size - patchH), (0, patch_size - patchW), (0, 0)), 'constant', constant_values=0)
            patch_list.append({
            'patch': patch,
            'start_h': beginH,
            'start_w': beginW,
            'end_h': endH,
            'end_w': endW
        })
    size = H_num*patch_size, W_num*patch_size
    return patch_list, im.shape[:2], size

def get_im_dct_qtb(patch, device = 'cpu',q=100):
    '''
    用于获取im和dct系数
    :param im:
    :return:
    '''
    with open('/data/jinrong/wcw/PS/DocTamper/qt_table.pk','rb') as fpk:
        pks = pickle.load(fpk)
    pkss = {}
    for k,v in pks.items():
        pkss[k] = torch.LongTensor(v)
    use_qtb = pkss[q]
    
    toctsr =torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225))
                                           ])
    
    with tempfile.NamedTemporaryFile(delete=True) as tmp:#创建临时文件
        im = Image.fromarray(patch)
        im = im.convert("L")#图像转为灰度图像
        im.save(tmp,"JPEG",quality=q)#按照质量因子压缩
        jpg = jpegio.read(tmp.name)
        dct = jpg.coef_arrays[0].copy()#获取图像的 DCT 系数
        im = im.convert('RGB')
        im = Image.fromarray(np.array(im))
#         im = torch.tensor(np.array(im))
        im = toctsr(im)
        dct_coef = torch.tensor(np.clip(np.abs(dct), 0,20))
#         dct_coef = torch.tensor(dct)
        use_qtb = pkss[q]
        im, dct_coef, use_qtb = Variable(im.to(device)), Variable(dct_coef.to(device)), Variable(
            use_qtb.unsqueeze(0).to(device))###
        im = im.unsqueeze(dim=0)
        dct_coef = dct_coef.unsqueeze(dim=0)
        use_qtb = use_qtb.unsqueeze(dim=0)
    return im,dct_coef,use_qtb

def get_patch_list(pic_path, device = 'cpu',q=100):
    #将原始图片变为512*512的patch_list
    patch_list, im_shape, paded_shape= get_patch(pic_path)
    im_dct_qtb_list = []
    for patch in patch_list:
        im, dct_coef, use_qtb = get_im_dct_qtb(patch['patch'], device = device,q=q)
        im_dct_qtb_list.append({
            'im': im,
            'dct': dct_coef,
            'use_qtb':use_qtb,
            'start_h': patch['start_h'],
            'start_w': patch['start_w'],
            'end_h': patch['end_h'],
            'end_w': patch['end_w']
        })
    return im_dct_qtb_list,im_shape,paded_shape

def zipdir(path, ziph):
    # ziph是zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            ziph.write(file_path, arcname=os.path.relpath(file_path, path))


#
if __name__=="__main__":
    with open('/data/jinrong/wcw/PS/DocTamper/qt_table.pk','rb') as fpk:
        pks = pickle.load(fpk)
    pkss = {}
    for k,v in pks.items():
        pkss[k] = torch.LongTensor(v)
    
    root_path = '/data/jinrong/wcw/PS/data/invoice/image/'
    file_path = [os.path.join(root_path,file) for file in os.listdir(root_path)]
    save_path0 = '/data/jinrong/wcw/PS/data/invoice_image_transnext_finetuing0722'
    os.makedirs(save_path0,exist_ok=True)
    save_path = [os.path.join(save_path0,file) for file in os.listdir(root_path)]
    
    model = seg_dtd()
    model.load_state_dict(torch.load('/data/jinrong/wcw/PS/DocTamper/train_logs/dtd_transNeXt/ckpt/2024-07-22-01/checkpoint-best.pth')['state_dict'])
    model = model.cuda()
    model.eval()
    for k,file in enumerate(file_path):
        im_dct_qtb_list,im_shape,paded_shape = get_patch_list(file,device = 'cuda',q=100)
        pred_mask = np.zeros(paded_shape, dtype=np.uint8)
        for data in tqdm.tqdm(im_dct_qtb_list):
            with torch.no_grad():
                pred = model(data['im'],data['dct'],data['use_qtb'])
                predt = torch.argmax(pred, dim=1)
            pred_mask[data['start_h']:data['end_h'],data['start_w']:data['end_w']] = predt.cpu().numpy()[0]
        pred_mask = pred_mask[:im_shape[0],:im_shape[1]]
        pred_mask_3 = cv2.cvtColor(pred_mask*255, cv2.COLOR_GRAY2BGR)
        img = cv2.imread(file)
        #分支一原始图片拼接
    #     img = np.concatenate([img, pred_mask_3], axis=1)
    #     cv2.imwrite(save_path[k],img)
        #分支二mask和原始图片堆叠
        mask_pixels = np.all(pred_mask_3==(255,255,255),axis=-1)
        pred_mask_3[mask_pixels] = (0, 204, 0)
        img_add = cv2.addWeighted(img, 0.8, pred_mask_3, 0.2, 0)
        img_add = np.clip(img_add, 0, 255).astype(np.uint8)
        #保存
        cv2.imwrite(save_path[k],img_add)

    # 创建一个ZipFile对象，mode参数设置为'w'表示写入模式
    zipf = zipfile.ZipFile(f'{save_path0}.zip', 'w', zipfile.ZIP_DEFLATED)

    # 将某个目录打包成zip文件
    if os.path.exists(save_path0):
        zipdir(save_path0, zipf)
    zipf.close()
    print('压缩成功')
    