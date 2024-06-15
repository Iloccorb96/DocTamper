import os
import time
import copy
import torch
import numpy as np
import torch.optim as optim
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import  DataLoader
from dataset.DocTamperDataset import DocTamperDataset
from utils.utils import AverageMeter  # , inial_logger
from .log import get_logger
from .metrics import IOUMetric
from torch.cuda.amp import autocast, GradScaler  # need pytorch>1.6
#引用上级目录中的losses文件夹下的函数
from ..loss import lovasz, soft_ce


Image.MAX_IMAGE_PIXELS = 1000000000000000


def train_net(param, model, device='cuda'):
    # 初始化参数

    train_imgs_dir = param['train_imgs_dir']
    train_labels_dir = param['train_labels_dir']
    val_imgs_dir = param['val_imgs_dir']
    val_labels_dir = param['val_labels_dir']

    model_name = param['model_name']
    epochs = param['epochs']
    batch_size = param['batch_size']
    iter_inter = param['iter_inter']
    save_log_dir = param['save_log_dir']
    save_ckpt_dir = param['save_ckpt_dir']
    load_ckpt_dir = param['load_ckpt_dir']
    save_epoch = param['save_epoch']
    T0 = param['T0']
    scaler = GradScaler()

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T0, T_mult=2, eta_min=1e-6, last_epoch=-1)

    LovaszLoss_fn = lovasz.LovaszLoss(mode='binary')#针对不平衡分类问题的损失函数
    SoftCrossEntropy_fn = soft_ce.SoftCrossEntropyLoss(smooth_factor=0.1)


    logger = get_logger(
        os.path.join(save_log_dir, time.strftime("%m-%d %H:%M:%S", time.localtime()) + '_' + model_name + '.log'))

    # 主循环
    train_loss_total_epochs, valid_loss_total_epochs, epoch_lr = [], [], []
    best_iou = 0
    best_mode = copy.deepcopy(model)
    epoch_start = 0
    #断点续训
    if load_ckpt_dir is not None:
        ckpt = torch.load(load_ckpt_dir)
        epoch_start = ckpt['epoch']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        q_level = ckpt['q_level']
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        #构建一个断点重新训练的数据集
        train_data = DocTamperDataset(train_imgs_dir, train_labels_dir, q_level=0, mode='train')
        valid_data = DocTamperDataset(val_imgs_dir, val_labels_dir, q_level=0, mode='val')
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4)
        valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False, num_workers=4)
        train_loader_size = train_loader.__len__()
        # valid_loader_size = valid_loader.__len__()

    CT_epoch = {0:0,3:0,6:1,9:2,15:3,21:4,33:5}
    for epoch in range(epoch_start, epochs):
        ## 配置课程学习策略,当余弦学习率周期调整时，调整样本压缩率
        if epoch in CT_epoch:
            q_level = CT_epoch[epoch]
            train_data = DocTamperDataset(train_imgs_dir, train_labels_dir, q_level,mode='train')
            valid_data = DocTamperDataset(val_imgs_dir, val_labels_dir, q_level,mode='val')

            train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4)
            valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False, num_workers=4)
            train_loader_size = train_loader.__len__()
            # valid_loader_size = valid_loader.__len__()

        epoch_start = time.time()
        # 训练阶段
        model.train()
        train_epoch_loss = AverageMeter()
        train_iter_loss = AverageMeter()
        for batch_idx, batch_samples in enumerate(train_loader):
            data, target, dct_coef, qs = batch_samples['image'], batch_samples['label'], batch_samples['dct'], batch_samples['qtb']#, batch_samples['q']
            data, target, dct_coef, qs = Variable(data.to(device)), Variable(target.to(device)), Variable(dct_coef.to(device)), Variable(qs.unsqueeze(1).to(device))
            optimizer.zero_grad()
            with autocast():  # need pytorch>1.6，混合精度训练
                pred = model(data,dct_coef,qs)
                loss = LovaszLoss_fn(pred, target) + SoftCrossEntropy_fn(pred, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            scheduler.step(epoch + batch_idx / train_loader_size)# 可以在每个批次结束后更新学习率，而不是等到整个 epoch 结束
            image_loss = loss.item()# 计算损失
            train_epoch_loss.update(image_loss)# 更新损失
            train_iter_loss.update(image_loss)# 更新损失
            if batch_idx % iter_inter == 0:#每隔一定数量的迭代打印一次日志信息
                spend_time = time.time() - epoch_start
                logger.info('[train] epoch:{} iter:{}/{} {:.2f}% lr:{:.6f} loss:{:.6f} ETA:{}min'.format(
                    epoch, #当前的训练轮次。
                    batch_idx,#当前处理的批次索引
                    train_loader_size,#训练数据加载器中的总批次数量。
                    batch_idx / train_loader_size * 100,#当前批次在整个epoch中的进度百分比。
                    optimizer.param_groups[-1]['lr'],#当前优化器的学习率。
                    train_iter_loss.avg,#到目前为止的平均损失值。
                    spend_time / (batch_idx + 1) * train_loader_size // 60 - spend_time // 60))#估计剩余时间（分钟）
                train_iter_loss.reset()#重置迭代损失值。

        # 验证阶段
        model.eval()
        valid_epoch_loss = AverageMeter()
        # valid_iter_loss = AverageMeter()
        iou = IOUMetric(2)
        with torch.no_grad():
            for batch_idx, batch_samples in enumerate(valid_loader):

                data, target, dct_coef, qs= batch_samples['image'], batch_samples['label'], batch_samples['dct'], batch_samples['qtb'], batch_samples['q']
                data, target, dct_coef, qs = Variable(data.to(device)), Variable(target.to(device)), Variable(dct_coef.to(device)), Variable(qs.unsqueeze(1).to(device))
                pred = model(data, dct_coef, qs)
                loss = LovaszLoss_fn(pred, target) + SoftCrossEntropy_fn(pred, target)
                pred = pred.cpu().data.numpy()
                pred = np.argmax(pred, axis=1)
                iou.add_batch(pred, target.cpu().data.numpy())
                image_loss = loss.item()
                valid_epoch_loss.update(image_loss)#只看epoch损失
            acc, acc_cls, iu, mean_iu, fwavacc, precision, recall, f1 = iou.evaluate()
            logger.info('[val] epoch:{} iou:{},acc:{}, precision:{},recall:{},f1:{}'.format(epoch, iu,acc,precision, recall, f1))

        # 保存loss、lr
        train_loss_total_epochs.append(train_epoch_loss.avg)
        valid_loss_total_epochs.append(valid_epoch_loss.avg)
        epoch_lr.append(optimizer.param_groups[0]['lr'])
        # 保存模型
        state = {'epoch': epoch,
                 'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'scheduler_state_dict': scheduler.state_dict() if scheduler else {},
                 'q_level':q_level}
        if epoch in save_epoch[T0]:
            torch.save(state,'{}/cosine_epoch{}.pth'.format(save_ckpt_dir, epoch))

        filename = os.path.join(save_ckpt_dir, 'checkpoint-latest.pth')
        torch.save(state, filename)  # pytorch1.6会压缩模型，低版本无法加载
        # 保存最优模型
        if iu > best_iou:  # train_loss_per_epoch valid_loss_per_epoch
            filename = os.path.join(save_ckpt_dir, 'checkpoint-best.pth')
            torch.save(state, filename)
            best_iou = iu
            best_mode = copy.deepcopy(model)
            logger.info('[save] Best Model saved at epoch:{} ============================='.format(epoch))

    return best_mode, model
#