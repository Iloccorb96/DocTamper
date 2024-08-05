import os
import time
import copy
import torch
import numpy as np
import torch.optim as optim
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset.DocTamperDataset import DocTamperDataset
from utils.utils import AverageMeter  # , inial_logger
from .log import get_logger
from .metrics import IOUMetric
from torch.cuda.amp import autocast, GradScaler  # need pytorch>1.6
# 引用上级目录中的losses文件夹下的函数
from loss import lovasz, soft_ce, OHEMSampler,ce,dice,pixelSCL
from transformers import get_cosine_schedule_with_warmup

torch.backends.cudnn.enabled=False


import random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
def worker_init_fn(worker_id):
    np.random.seed(SEED + worker_id)



Image.MAX_IMAGE_PIXELS = 1000000000000000



def train_net(param, model, device='cuda'):
    # 初始化参数

    
    train_image_list = param['train_image_list']
    test_image_list = param['test_image_list']
    train_label_list = param['train_label_list']
    test_label_list = param['test_label_list']
    
    model_name = param['model_name']
    
    batch_size = param['batch_size']
    iter_inter = param['iter_inter']
    save_log_dir = param['save_log_dir']
    save_ckpt_dir = param['save_ckpt_dir']
    load_ckpt_dir = param['load_ckpt_dir']
    save_epoch = param['save_epoch']
    
    epochs = param['epochs']
    warm_up = param['warm_up']
    scaler = GradScaler()

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-4)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warm_up, num_training_steps=epochs,
                                                num_cycles=0.5)
    #损失函数
    LovaszLoss_fn = lovasz.LovaszLoss(mode='multiclass')  # 针对不平衡分类问题的损失函数
    dice_fn = dice.DiceLoss(mode='multiclass')
    SoftCrossEntropy_fn = soft_ce.SoftCrossEntropyLoss(smooth_factor=0.1)
    pixelSCL_fn = pixelSCL.PixelContrastLoss()
#     weight = torch.tensor([1.0,10.0]).to('cuda')
#     CE_fn = ce.CrossEntropyLoss(weight=weight)
    # OHEM采样器
    ohem_sampler = OHEMSampler(thresh=0.5, min_kept=100000)
    
    logger = get_logger(
        os.path.join(save_log_dir, time.strftime("%m-%d %H:%M:%S", time.localtime()) + '_' + model_name + '.log'))

    # 主循环
    train_loss_total_epochs, valid_loss_total_epochs, epoch_lr = [], [], []
    best_iou,epoch_start = 0,0
    best_model = copy.deepcopy(model)
    
    #固定验证集
    valid_data = DocTamperDataset(test_image_list, test_label_list, q_level=0, mode='val')
    valid_loader = DataLoader(dataset=valid_data, batch_size=2, shuffle=False, num_workers=4,worker_init_fn=worker_init_fn)
    
    trained_iter = None
    # 仅加载预训练参数或者断点续训
    if load_ckpt_dir is not None:
        print('loading checkpoints....')
        ckpt = torch.load(load_ckpt_dir)
        if 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'])
        else:
            model.load_state_dict(ckpt)
        if 'q_level' in ckpt:
            q_level = ckpt['q_level']
        if 'epoch' in ckpt:
            epoch_start = ckpt['epoch']+1
            # 构建一个断点重新训练的数据集
            train_data = DocTamperDataset(train_image_list, train_label_list, q_level=q_level, mode='train')
            train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4)
            train_loader_size = train_loader.__len__()
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    CT_epoch = {0:0, 2:0,  12:1,  22:2,  32:3, 52:4,  72:5}
    training_stopped = False
    

    for epoch in range(epoch_start, epochs):
        if training_stopped:
            break
        ## 配置课程学习策略,当余弦学习率周期调整时，调整样本压缩率
        if epoch in CT_epoch:
            q_level = CT_epoch[epoch]
            train_data = DocTamperDataset(train_image_list, train_label_list, q_level, mode='train')
#             valid_data = DocTamperDataset(test_image_list, test_label_list, q_level=5, mode='val')
            train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4,worker_init_fn=worker_init_fn)
#             valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False, num_workers=4,worker_init_fn=worker_init_fn)
            train_loader_size = train_loader.__len__()
            # valid_loader_size = valid_loader.__len__()
        trained_iter = epoch*train_loader_size
        train_start = time.time()
        # 训练阶段
        model.train()
        train_epoch_loss = AverageMeter()
        train_iter_loss = AverageMeter()
        for batch_idx, batch_samples in enumerate(train_loader):
            try:
                iou_train = IOUMetric(2)
                data, target, dct_coef, qs = batch_samples['image'], batch_samples['label'], batch_samples['dct'], batch_samples['qtb']
                data, target, dct_coef, qs = Variable(data.to(device)), Variable(target.to(device)), Variable(dct_coef.to(device)), Variable(qs.unsqueeze(1).to(device))
                assert torch.isfinite(data).all().item() , "Data contains NaN or Inf values"
                assert torch.isfinite(target).all().item(), "Target contains NaN or Inf values"
                optimizer.zero_grad()
                with autocast():
                    pred,embedding = model(data, dct_coef, qs)
                    iou_train.add_batch(pred.argmax(dim=1).cpu().data.numpy(), target.cpu().data.numpy())
                    pixelSCL_fn._dequeue_and_enqueue(embedding.float(), target,
                                      segment_queue_x=model.model.segment_queue_x,
                                      segment_queue_y=model.model.segment_queue_y ,
                                      segment_queue_ptr=model.model.segment_queue_ptr ,
                                      pixel_queue_x=model.model.pixel_queue_x,
                                      pixel_queue_y=model.model.pixel_queue_y ,
                                      pixel_queue_ptr=model.model.pixel_queue_ptr ,)
                    queue_X = torch.concat((model.model.segment_queue_x,model.model.pixel_queue_x),dim=0)
                    queue_y = torch.concat((model.model.segment_queue_y,model.model.pixel_queue_ptr),dim=0)
                    queue = (queue_X,queue_y)
                    pixel_loss = pixelSCL_fn(embedding.float(),target,pred,queue)
                    seg_loss = dice_fn(pred, target) + SoftCrossEntropy_fn(pred, target)
                    # loss = CE_fn(pred, target)+LovaszLoss_fn(pred, target)
                    # OHEM采样
                    selected_mask = ohem_sampler(pred, target)
                    sampled_losses = seg_loss * selected_mask
                    loss_ohem = torch.mean(sampled_losses)

                    loss = loss_ohem+0.1*pixel_loss
                    # Check if loss is NaN
                    if torch.isnan(pred).any().item():
                        raise ValueError("pred is NaN")
                    if torch.isnan(loss).item():
                        raise ValueError("Loss is NaN")

                    #autocast should wrap only the forward pass(es) of your network, including the loss computation(s).
                    #Backward passes under autocast are not recommended.
                scaler.scale(loss).backward()

#                     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()

                scheduler.step(epoch + batch_idx / len(train_loader))# 可以在每个批次结束后更新学习率，而不是等到整个 epoch 结束
                image_loss = loss.item()  # 计算损失
                train_epoch_loss.update(image_loss)  # 更新损失
                train_iter_loss.update(image_loss)  # 更新损失
                trained_iter += 1
                if trained_iter % iter_inter == 0 :  # 每隔一定数量的迭代打印一次日志信息
                    spend_time = time.time() - train_start
                    acc, acc_cls, iu, mean_iu, fwavacc, precision, recall, f1 = iou_train.evaluate()
                    logger.info('[train] epoch:{} now_iter/total_iter:{}/{} {:.2f}% lr:{:.6f} loss:{:.6f},now_iou:{:.4f},now_f1:{:.6f}, ETA:{}min'.format(
                        epoch, #当前的训练轮次。
                        trained_iter,#当前处理的批次索引
                        train_loader_size*epochs, #总iter数
#                             train_loader_size,#训练数据加载器中的总批次数量。
                        batch_idx / train_loader_size * 100,#当前批次在整个epoch中的进度百分比。
                        optimizer.param_groups[-1]['lr'],#当前优化器的学习率。
                        train_iter_loss.avg,#到目前为止的平均损失值。
                        iu[1],
                        f1[1],
                        spend_time / (batch_idx + 1) * train_loader_size // 60 - spend_time // 60))#估计剩余时间（分钟）
                    train_iter_loss.reset()#重置迭代损失值。
                    iou_train = IOUMetric(2)
            except Exception as e:
                # Save relevant information for debugging
                torch.save({
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'pred': pred,
                    'loss':loss,
                    'data': data.cpu(),
                    'target': target.cpu(),
                    'dct_coef': dct_coef.cpu(),
                    'qs': qs.cpu(),
                    'trained_iter':trained_iter,
                }, f"error_checkpoint_epoch_{epoch}_batch_{batch_idx}.pt")
                print(f"Error occurred at epoch {epoch}, batch {batch_idx}: {e}")



        # 验证阶段
        model.eval()
        valid_epoch_loss = AverageMeter()
        # valid_iter_loss = AverageMeter()
        iou = IOUMetric(2)
        with torch.no_grad():
            for batch_idx, batch_samples in enumerate(valid_loader):
                data, target, dct_coef, qs = batch_samples['image'], batch_samples['label'], batch_samples['dct'], \
                batch_samples['qtb']
                data, target, dct_coef, qs = Variable(data.to(device)), Variable(target.to(device)), Variable(
                    dct_coef.to(device)), Variable(qs.unsqueeze(1).to(device))
                pred = model(data, dct_coef, qs)
                loss = LovaszLoss_fn(pred, target) + SoftCrossEntropy_fn(pred, target)
                iou.add_batch(pred.argmax(dim=1).cpu().data.numpy(), target.cpu().data.numpy())
                image_loss = loss.item()
                valid_epoch_loss.update(image_loss)  # 只看epoch损失
            acc, acc_cls, iu, mean_iu, fwavacc, precision, recall, f1 = iou.evaluate()
            print(f'iu:{iu}')
            logger.info(
                '[val] epoch:{} iou:{},acc:{}, precision:{},recall:{},f1:{}'.format(epoch, iu, acc, precision, recall,f1))
        # 保存loss、lr
        train_loss_total_epochs.append(train_epoch_loss.avg)
        valid_loss_total_epochs.append(valid_epoch_loss.avg)
        epoch_lr.append(optimizer.param_groups[0]['lr'])
        # 保存模型
        state = {'epoch': epoch,
                 'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'scheduler_state_dict': scheduler.state_dict() if scheduler else {},
                 'q_level': q_level}
        if epoch in save_epoch:
            torch.save(state, '{}/cosine_epoch{}.pth'.format(save_ckpt_dir, epoch))

        filename = os.path.join(save_ckpt_dir, 'checkpoint-latest.pth')
        torch.save(state, filename)  # pytorch1.6会压缩模型，低版本无法加载
        # 保存最优模型
        if iu[1] > best_iou:
            filename = os.path.join(save_ckpt_dir, 'checkpoint-best.pth')
            torch.save(state, filename)
            best_iou = iu[1]
            logger.info('[save] Best Model saved at epoch:{} ============================='.format(epoch))
            filename = os.path.join(save_ckpt_dir, 'checkpoint-best.pth')
            torch.save(state, filename)
