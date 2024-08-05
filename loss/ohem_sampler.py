import torch
import torch.nn as nn
import torch.nn.functional as F

class OHEMSampler(nn.Module):
    def __init__(self, thresh, min_kept, ignore_index=-100):
        super(OHEMSampler, self).__init__()
        self.thresh = thresh
        self.min_kept = min_kept
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # 假设logits是模型输出的logits，targets是真实标签
        # OHEM 采样过程
        batch_size, num_classes, height, width = logits.size()
        ##targets的尺寸：batch_size, height, width

        # 将logits转换为概率
        probs = F.softmax(logits, dim=1)

        # 计算每个像素的损失
        mask = targets != self.ignore_index
        #像素点对应label类别的概率越大，即损失越小，通过-log实现交
        log_probs = torch.log(probs + 1e-7)
        losses = -torch.gather(log_probs, 1,targets.unsqueeze(1)).squeeze(1) * mask.float()

        ## 将损失展平为二维张量，准备排序
        losses_flatten = losses.view(batch_size, -1)  # 形状变为 [batch_size, height * width]
        # 将损失按像素排序
        sorted_losses, _ = losses_flatten.sort(dim=1)  # 对每个batch中的像素损失进行排序

        # 计算阈值
        # 由于sorted_losses已经是展平后的二维张量，我们需要先计算每个batch的阈值
        num_pixels = sorted_losses.size(1)  # 每个batch中的像素总数
        # 使用min函数确保不会超过像素总数
        thresh_indices = max(self.min_kept, int(num_pixels * self.thresh))
        thresh_indices = min(thresh_indices, num_pixels - 1)  # 防止索引越界

        thresh = sorted_losses[:, thresh_indices]  # 获取阈值

        # 选择难例
        selected_mask = losses_flatten > thresh.unsqueeze(1)  # 扩展阈值维度以进行比较
        # 确保真实标签（正样本）被选中，默认0（负样本除了困难样本不会被置为1）
        for i in range(batch_size):
            selected_mask[i, targets[i].flatten()] = 1

        # 将选中的像素mask恢复到原始尺寸
        selected_mask = selected_mask.view_as(losses)


        selected_mask = selected_mask.float()

        # 返回采样后mask矩阵
        return selected_mask