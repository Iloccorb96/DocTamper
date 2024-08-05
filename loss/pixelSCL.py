import torch
import torch.nn as nn
import torch.nn.functional as F
# import pdb

class PixelContrastLoss(nn.Module):
    def __init__(self):
        super(PixelContrastLoss, self).__init__()
        self.temperature = 1  # 0.07-0.1
        self.base_temperature = 1  # 0.07-0.1
        self.ignore_label = -1
        #         self.pmin = 2
        self.pmax = 500
        self.memory_size = 10000
        self.pixel_update_freq = 10

    def _smi_hard_anchor_sampling(self, X, y_hat, y):
        '''
        # X: 输入的特征图，形状为 [batch_size, height,width, feat_dim]
        # y_hat: 预测的标签，形状为 [batch_size, height, width]
        # y: 真实的标签，形状为 [batch_size, height, width]
        # X_:[n,F]
        # y_:[n]
        '''
        feat_dim = X.shape[-1]

        X = X.contiguous().view(-1, feat_dim)
        y_hat = y_hat.contiguous().view(-1)
        y = y.contiguous().view(-1)


        num_tamp = (y == 1).nonzero().shape[0]
        num_auth = (y == 0).nonzero().shape[0]
        # 1确定tamp难例和易例的数量
        hard_tamp_indices = ((y == 1) & (y_hat != y)).nonzero()
        easy_tamp_indices = ((y == 1) & (y_hat == y)).nonzero()
        num_tamp_hard = hard_tamp_indices.shape[0]
        num_tamp_easy = easy_tamp_indices.shape[0]
        if num_tamp_hard>0 and num_tamp_easy>0:
            num_tamp_hard_keep = int(min(min(num_tamp, self.pmax) / 2, num_tamp_hard))
            num_tamp_easy_keep = int(min(min(num_tamp, self.pmax) / 2, num_tamp_easy))
            perm = torch.randperm(num_tamp_hard)  # 将12345打乱
            hard_tamp_indices = hard_tamp_indices[perm[:num_tamp_hard_keep]]  # 随机取num_hard_keep个难例
            perm = torch.randperm(num_tamp_easy)
            easy_tamp_indices = easy_tamp_indices[perm[:num_tamp_easy_keep]]  # 随机取num_easy个简例
            tamp_indices = torch.cat((hard_tamp_indices,easy_tamp_indices))
        elif num_tamp_hard>0:
            num_tamp_hard_keep = int(min(min(num_tamp, self.pmax) / 2, num_tamp_hard))
            perm = torch.randperm(num_tamp_hard)
            hard_tamp_indices = hard_tamp_indices[perm[:num_tamp_hard_keep]]
            tamp_indices = hard_tamp_indices
        else:
            num_tamp_easy_keep = int(min(min(num_tamp, self.pmax) / 2, num_tamp_easy))
            perm = torch.randperm(num_tamp_easy)
            easy_tamp_indices = easy_tamp_indices[perm[:num_tamp_easy_keep]]
            tamp_indices = easy_tamp_indices
        # 2确定auth难例和易例的数量
        hard_auth_indices = ((y == 0) & (y_hat != y)).nonzero()
        easy_auth_indices = ((y == 0) & (y_hat == y)).nonzero()
        num_auth_hard = hard_auth_indices.shape[0]
        num_auth_easy = easy_auth_indices.shape[0]
        if num_auth_hard>0 and num_auth_easy>0:
            num_auth_hard_keep = int(min(min(num_auth,self.pmax) / 2, num_auth_hard))
            num_auth_easy_keep = int(min(min(num_auth, self.pmax) / 2, num_auth_easy))
            perm = torch.randperm(num_auth_hard)  # 将12345打乱
            hard_auth_indices = hard_auth_indices[perm[:num_auth_hard_keep]]  # 随机取num_hard_keep个难例
            perm = torch.randperm(num_auth_easy)
            easy_auth_indices = easy_auth_indices[perm[:num_auth_easy_keep]]  # 随机取num_easy个简例
            auth_indices = torch.cat((hard_auth_indices,easy_auth_indices))
        elif num_auth_hard>0:
            num_auth_hard_keep = int(min(min(num_auth, self.pmax) / 2, num_auth_hard))
            perm = torch.randperm(num_auth_hard)
            hard_auth_indices = hard_auth_indices[perm[:num_auth_hard_keep]]
            auth_indices = hard_auth_indices
        else:
            num_auth_easy_keep = int(min(min(num_auth, self.pmax) / 2, num_auth_easy))
            perm = torch.randperm(num_auth_easy)
            easy_auth_indices = easy_auth_indices[perm[:num_auth_easy_keep]]
            auth_indices = easy_auth_indices
        
        index = torch.cat((tamp_indices,auth_indices))
        
        index = index.squeeze()
        X_ = X[index, :]
        y_ = y[index]
        return X_, y_


    def _sample_negative(self, Q):
        '''
        Args:
            Q: 【n, feat_size】

        Returns:X_:n, feat_size
        y_:n

        '''
        queue_X,queue_y = Q
        index = ((queue_y==1)|(queue_y==0)).nonzero()
        
        # 这里一开始迭代的时候，队列后n行会是随机初始化的向量
        X_ = queue_X[index,:].squeeze()
        y_ = queue_y[index].squeeze()
        
        return X_, y_


    def _contrastive(self, X_anchor, y_anchor, queue=None):
        # X_anchor 是锚点特征，形状为 [n, feature_dim]
        # y_anchor 是锚点标签，形状为 [n]
        # queue 是可选的，用于存储负样本的特征队列

        # 样本数量
        n = X_anchor.shape[0]
        y_anchor = y_anchor.contiguous().view(-1, 1)

        if queue is not None:  
            X_contrast, y_contrast = self._sample_negative(queue)#X_contrast[n, feat_size]
            y_contrast = y_contrast.contiguous().view(-1, 1)
            mask = torch.eq(y_anchor, y_contrast.T).float().cuda()
            
        else:
            y_contrast = y_anchor 
            X_contrast = X_anchor 
            mask = torch.eq(y_anchor, y_contrast.T).float().cuda()
            mask = mask * (1 - torch.eye(n, dtype=torch.float).cuda())

        # 计算锚点特征和负样本特征的点积，然后除以温度参数，形状为 [anchor_num * n_view=B*C*pixel, contrast_count],如800*800
        anchor_dot_contrast = torch.div(torch.matmul(X_anchor, X_contrast.T),
                                        self.temperature)
        
        # 计算每个样本的最大 logits，用于稳定性
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # 从每个锚点的 logits 中减去最大值，以提高数值稳定性
        logits = anchor_dot_contrast - logits_max.detach()
        
        
        neg_mask = 1 - mask  # neg样本

        neg_logits = torch.exp(logits) * neg_mask  # 计算负样本的指数
        neg_logits = neg_logits.sum(1, keepdim=True)  # 对负样本的指数求和，按行即按样本求和
        # 计算对数概率
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits + neg_logits)  # 对应公式的log处

        # 计算正样本的对数概率的均值
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # 计算对比损失
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()  # 计算损失的均值

        return loss

    def forward(self, feats, labels=None, predict=None, queue=None):
        '''
        Args:
            feats: embedding [B,F,H,W]
            labels: [B,H,W]
            predict:[B,C,H,W]
            queue:

        Returns:

        '''
        ## 将标签 labels 重塑为 [batch_size, 1]，并复制一份以避免修改原始数据
        labels = labels.unsqueeze(1).clone()
        ## 将标签 labels 下采样到与特征图 feats 相同的空间分辨率，使用最近邻插值
        labels = torch.nn.functional.interpolate(labels.float(), (feats.shape[2], feats.shape[3]), mode='nearest')
        # 将上采样后的标签从 [batch_size, 1, height, width] 重塑为 [batch_size, height, width]
        labels = labels.squeeze(1).long()

        predict = predict.argmax(dim=1)
        predict = torch.nn.functional.interpolate(predict.float().unsqueeze(1), (feats.shape[2], feats.shape[3]), mode='nearest').squeeze(1)
        # 确保标签的形状与特征图的宽度一致，否则抛出错误
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

        batch_size = feats.shape[0]

        labels = labels.contiguous().view(batch_size, -1)  # 展平为B H*W
        predict = predict.contiguous().view(batch_size, -1)  # 展平为B H*W
        feats = feats.permute(0, 2, 3, 1)  # B H W C
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])  # B H*W C
#         print(feats.shape,labels.shape)
#         pdb.set_trace()
        feats_, labels_ = self._smi_hard_anchor_sampling(feats, predict, labels)
#         print(feats_.shape,labels_.shape)
        loss = self._contrastive(feats_, labels_, queue=queue)
        return loss

    def _dequeue_and_enqueue(self, embedding, labels,
                             segment_queue_x, segment_queue_y,segment_queue_ptr,
                             pixel_queue_x, pixel_queue_y, pixel_queue_ptr):
        '''

        Args:
            self:每个label的memory size是5000
            embedding: embedding，【B,F,H，W】
            labels: [B,H,W]
            segment_queue:(10000, 64)
            segment_queue_ptr:(10000)
            pixel_queue: (10000, 64)
            pixel_queue_ptr:(10000)

        Returns:

        '''
        
        batch_size = embedding.shape[0]
        feat_dim = embedding.shape[1]

        labels = F.interpolate(labels.float().unsqueeze(1), scale_factor=0.5, mode='nearest').squeeze(1)

        for bs in range(batch_size):
            
            this_feat = embedding[bs].permute(1,2,0).contiguous().view(-1,64)  # [F,H,W]->[H*W,F]
            this_label = labels[bs].contiguous().view(-1)  # 【H,W】->【H*W】
            
            tamp_indices = (this_label == 1).nonzero().squeeze()  # 找到标签为 1 的索引
            auth_indices = (this_label == 0).nonzero().squeeze()  # 找到标签为 0 的索引
            #pixel
            sample_tamp = tamp_indices[torch.randperm(tamp_indices.size(0))[:10]]
            sample_auth = auth_indices[torch.randperm(auth_indices.size(0))[:10]]
            sampled_index = torch.cat((sample_tamp,sample_auth),dim=0)
            
            num_index = sampled_index.shape[0]
            if pixel_queue_ptr + num_index >= self.memory_size:
                pixel_queue_x[-num_index:, :] = nn.functional.normalize(this_feat[sampled_index], p=2, dim=1)
                pixel_queue_y[-num_index:] = this_label[sampled_index]
                pixel_queue_ptr = 0
            else:
                pixel_queue_x[pixel_queue_ptr:pixel_queue_ptr + num_index, :] = nn.functional.normalize(this_feat[sampled_index], p=2, dim=1)
                pixel_queue_y[pixel_queue_ptr:pixel_queue_ptr + num_index] = this_label[sampled_index]
                pixel_queue_ptr = (pixel_queue_ptr + 1) % self.memory_size  
                
            # seg
            feat_tamp = torch.mean(this_feat[tamp_indices,], dim=0).unsqueeze(0)
            feat_auth = torch.mean(this_feat[auth_indices,], dim=0).unsqueeze(0)
            feat = torch.cat((feat_tamp,feat_auth),dim=0)
            if segment_queue_ptr + 2 >= self.memory_size:
                segment_queue_x[-2:, :] = nn.functional.normalize(feat, p=2, dim=1)
                segment_queue_y[-2:] = torch.tensor([1,0])
                segment_queue_ptr = 0
            else:
                segment_queue_x[segment_queue_ptr:segment_queue_ptr + 2, :] = nn.functional.normalize(feat, p=2, dim=1)
                segment_queue_y[segment_queue_ptr:segment_queue_ptr + 2] = torch.tensor([1,0])
                segment_queue_ptr = (segment_queue_ptr + 2) % self.memory_size  