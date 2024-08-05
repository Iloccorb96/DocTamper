import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
import math

#modal SRM
class SRMFilter(nn.Module):
    def __init__(self):
        super().__init__()
        f1 = [[0, 0, 0, 0, 0],
              [0, -1, 2, -1, 0],
              [0, 2, -4, 2, 0],
              [0, -1, 2, -1, 0],
              [0, 0, 0, 0, 0]]

        f2 = [[-1, 2, -2, 2, -1],
              [2, -6, 8, -6, 2],
              [-2, 8, -12, 8, -2],
              [2, -6, 8, -6, 2],
              [-1, 2, -2, 2, -1]]

        f3 = [[0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0],
              [0, 1, -2, 1, 0],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0]]

        q = torch.tensor([[4.], [12.], [2.]]).unsqueeze(-1).unsqueeze(-1)
        filters = torch.tensor([[f1, f1, f1], [f2, f2, f2], [f3, f3, f3]], dtype=torch.float) / q #卷积核的形状为【outc,inc,h,w】
        self.register_buffer('filters', filters)#使用 register_buffer 方法将滤波器张量 filters 注册为模块的一个缓冲区，缓冲区不会在训练过程中被优化：
        self.truc = nn.Hardtanh(-2, 2)#定义了一个 Hardtanh 激活函数，用于对卷积结果进行激活，范围在 [-2, 2] 之间：

    def forward(self, x):
        x = F.conv2d(x, self.filters, padding=2, stride=1)
        x = self.truc(x)
        return x
    
    
#modal BayarConv
class BayarConv2d(nn.Module):
    '''
    "Bayar 约束"。
    这种约束在卷积核的中心元素固定为 -1，
    而其他元素是可训练的参数，
    并且在每次前向传播时保持卷积核的权重和为 1。
    '''
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.minus1 = (torch.ones(self.in_channels, self.out_channels, 1) * -1.000)

        super().__init__()
        # only (kernel_size ** 2 - 1) trainable params as the center element is always -1
        self.kernel = nn.Parameter(torch.rand(self.in_channels, self.out_channels, kernel_size ** 2 - 1),
                                   requires_grad=True)#[inc,outc,size^2-1]

    def bayarConstraint(self):
        self.kernel.data = self.kernel.data.div(self.kernel.data.sum(-1, keepdims=True))#卷积核的权重进行归一化，使每个卷积核的权重和为 1
        ctr = self.kernel_size ** 2 // 2 #上下个半，size5则为12
        real_kernel = torch.cat((self.kernel[:, :, :ctr], self.minus1.to(self.kernel.device), self.kernel[:, :, ctr:]), dim=2)#拼成size**2
        real_kernel = real_kernel.reshape((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        return real_kernel

    def forward(self, x):
        x = F.conv2d(x, self.bayarConstraint(), stride=self.stride, padding=self.padding)
        return x
    
#modal noiseprint++
def conv_with_padding(in_planes, out_planes, kernelsize, stride=1, dilation=1, bias=False, padding = None):
    if padding is None:
        padding = kernelsize//2
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernelsize, stride=stride, dilation=dilation, padding=padding, bias=bias)

def conv_init(conv, act='linear'):
    r"""
    Reproduces conv initialization from DnCNN
    """
    n = conv.kernel_size[0] * conv.kernel_size[1] * conv.out_channels
    conv.weight.data.normal_(0, math.sqrt(2. / n))

def batchnorm_init(m, kernelsize=3):
    r"""
    Reproduces batchnorm initialization from DnCNN
    """
    n = kernelsize**2 * m.num_features
    m.weight.data.normal_(0, math.sqrt(2. / (n)))
    m.bias.data.zero_()

def make_activation(act):
    if act is None:
        return None
    elif act == 'relu':
        return nn.ReLU(inplace=True)
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'leaky_relu':
        return nn.LeakyReLU(inplace=True)
    elif act == 'softmax':
        return nn.Softmax()
    elif act == 'linear':
        return None
    else:
        assert(False)

def make_net(nplanes_in, kernels, features, bns, acts, dilats, bn_momentum = 0.1, padding=None):
    r"""
    :param nplanes_in: number of of input feature channels
    :param kernels: list of kernel size for convolution layers
    :param features: list of hidden layer feature channels
    :param bns: list of whether to add batchnorm layers
    :param acts: list of activations
    :param dilats: list of dilation factors
    :param bn_momentum: momentum of batchnorm
    :param padding: integer for padding (None for same padding)
    """

    depth = len(features)
    assert(len(features)==len(kernels))

    layers = list()
    for i in range(0,depth):
        if i==0:
            in_feats = nplanes_in
        else:
            in_feats = features[i-1]

        elem = conv_with_padding(in_feats, features[i], kernelsize=kernels[i], dilation=dilats[i], padding=padding, bias=not(bns[i]))
        conv_init(elem, act=acts[i])
        layers.append(elem)

        if bns[i]:
            elem = nn.BatchNorm2d(features[i], momentum = bn_momentum)
            batchnorm_init(elem, kernelsize=kernels[i])
            layers.append(elem)

        elem = make_activation(acts[i])
        if elem is not None:
            layers.append(elem)

    return nn.Sequential(*layers)


class ModalitiesExtractor(nn.Module):
    def __init__(self,
                 modals: list = ('noiseprint', 'bayar', 'srm'),
                 noiseprint_path: str = None):
        super().__init__()
        self.mod_extract = []
        if 'noiseprint' in modals:
            num_levels = 17
            out_channel = 1
            self.noiseprint = make_net(3, kernels=[3, ] * num_levels,
                                  features=[64, ] * (num_levels - 1) + [out_channel],
                                  bns=[False, ] + [True, ] * (num_levels - 2) + [False, ],
                                  acts=['relu', ] * (num_levels - 1) + ['linear', ],
                                  dilats=[1, ] * num_levels,
                                  bn_momentum=0.1, padding=1)

            if noiseprint_path:
                np_weights = noiseprint_path
                assert os.path.isfile(np_weights)
                dat = torch.load(np_weights, map_location=torch.device('cpu'))
#                 logging.info(f'Noiseprint++ weights: {np_weights}')
                self.noiseprint.load_state_dict(dat)

            self.noiseprint.eval()
            for param in self.noiseprint.parameters():
                param.requires_grad = False
            self.mod_extract.append(self.noiseprint)
        if 'bayar' in modals:
            self.bayar = BayarConv2d(3, 3, padding=2)
            self.mod_extract.append(self.bayar)
        if 'srm' in modals:
            self.srm = SRMFilter()
            self.mod_extract.append(self.srm)

    def set_train(self):
        if hasattr(self, 'bayar'):
            self.bayar.train()

    def set_val(self):
        if hasattr(self, 'bayar'):
            self.bayar.eval()

    def forward(self, x) -> list:
        out = []
        for mod in self.mod_extract:
            y = mod(x)
            if y.size()[-3] == 1:
                y = torch.tile(y, (3, 1, 1))
            out.append(y)

        return out
