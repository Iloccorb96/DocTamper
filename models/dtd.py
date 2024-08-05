from torch.cuda.amp import autocast, GradScaler#need pytorch>1.6
import sys
sys.path.append('/data/jinrong/wcw/PS/DocTamper/models')
from fph import FPH
from swins import *
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from functools import partial
from segmentation_models_pytorch.base import modules as md
from segmentation_models_pytorch.base import SegmentationModel

class LayerNorm(nn.Module):
    # LayerNorm
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        #normalized_shape是输入通道数
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":#Channels Last (NHWC)
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":#Channels First (NCHW)
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):#reduction这允许网络更加关注于重要的特征，同时减少计算量。
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)

class ConvBlock(nn.Module):
    # dwconv + norm + pwconv1 + act + pwconv2 + gamma
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)#深度可分离卷积层
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)#全连接层，用于改变特征图的通道数。
        self.act = nn.GELU()#将逐元素地应用在输入张量上，因此输入形状和输出形状保持一致。
        self.pwconv2 = nn.Linear(4 * dim, dim)#全连接层，用于改变特征图的通道数。
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        ipt = x
        x = self.dwconv(x)#B C H W
        x = x.permute(0, 2, 3, 1)#B H W C permute(0, 2, 3, 1)将通道数移到最后一维，方便在layernorm中使用chanel_last的方法以及方便全连接层的输入，
        x = self.norm(x)
        x = self.pwconv1(x)#输入形状应该是 (batch_size, *, input_features)。
        x = self.act(x)
        x = self.pwconv2(x)#输入形状应该是 (batch_size, *, input_features)。
        if self.gamma is not None:
            x = self.gamma * x #B H W C 可学习的缩放参数 gamma，用于在模型的每一层之后缩放输出，从而控制其幅度，较大的初始值可能导致梯度爆炸或梯度消失问题。随着训练的进行，gamma 会被梯度下降算法逐渐调整到合适的值，从而逐步增加每一层输出的幅度。
        x = x.permute(0, 3, 1, 2)#permute(0, 3, 1, 2)将通道数移到第一维，方便后续操作。
        x = ipt + self.drop_path(x)
        return x

class AddCoords(nn.Module):
    def __init__(self, with_r=True):
        super().__init__()
        self.with_r = with_r
    def forward(self, input_tensor):
        batch_size, _, x_dim, y_dim = input_tensor.size()
        # 函数生成二维网格坐标，对应于特征图的宽度（x轴）和高度（y轴）。xx_c和yy_c分别代表了y轴和x轴方向上的坐标矩阵。
        # 0,1,2...size-1
        xx_c, yy_c = torch.meshgrid(torch.arange(x_dim,dtype=input_tensor.dtype), torch.arange(y_dim,dtype=input_tensor.dtype))
        #将坐标值归一化到[-1, 1]区间内，这是常见的归一化处理，有助于模型学习，使得坐标值与输入特征图的数值范围保持一致
        xx_c = xx_c.to(input_tensor.device) / (x_dim - 1) * 2 - 1
        yy_c = yy_c.to(input_tensor.device) / (y_dim - 1) * 2 - 1
        #使用.expand方法将单通道的坐标矩阵复制扩展到与输入特征图相同的batch size和通道数（最初为1，因为只有坐标信息），使得每个样本的每个位置都有相应的坐标值。
        xx_c = xx_c.expand(batch_size,1,x_dim,y_dim)
        yy_c = yy_c.expand(batch_size,1,x_dim,y_dim)
        ret = torch.cat((input_tensor,xx_c,yy_c), dim=1)
        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_c - 0.5, 2) + torch.pow(yy_c - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)
        return ret#shape[batch_size, channels+2+1, x_dim, y_dim]

class VPH(nn.Module):
    def __init__(self, dims=[96, 192,384,768], drop_path_rate=0.4, layer_scale_init_value=1e-6):
        super().__init__()
        self.dims = dims
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, 6)]#6个dropout率，等差数列
        self.downsample_layers = nn.ModuleList([nn.Sequential(nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
                                                              LayerNorm(dims[0], eps=1e-6, data_format="channels_first")),
                                                nn.Sequential(LayerNorm(dims[1], eps=1e-6, data_format="channels_first"),
                                                              nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2))])
        self.stages = nn.ModuleList([nn.Sequential(*[ConvBlock(dim=dims[0], drop_path=dp_rates[j],
                                                               layer_scale_init_value=layer_scale_init_value) for j in range(3)]),#layer_scale_init_value使得每层的输出的幅度变化更加平滑，避免梯度爆炸或梯度消失问题。
                                     nn.Sequential(*[ConvBlock(dim=dims[1], drop_path=dp_rates[3 + j],
                                                               layer_scale_init_value=layer_scale_init_value) for j in range(3)])])
        #stage:3个convblock
        self.apply(self._init_weights)#对模型中的每一层应用 _init_weights 方法，用于初始化每一层的权重。
        self.initnorm()#初始化4个layernorm层，可以通过类似 self.norm0 的方式进行访问。

    def initnorm(self):
        #初始化模型中的 LayerNorm层。
        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")## 创建LayerNorm层的实例，其中eps和data_format参数被固定。partial函数用于创建一个新的函数，该函数在调用时会带有预设的参数值。
        for i_layer in range(4):
            layer = norm_layer(self.dims[i_layer])#部分函数实例化 LayerNorm 层，self.dims[i_layer] 作为 normalized_shape 传递。
            layer_name = f'norm{i_layer}'# 构建LayerNorm层的名称，例如"norm0", "norm1", ..., "norm3"
            self.add_module(layer_name, layer)# add_module是Module类的一个方法，用于将子模块添加到模块中。

    def _init_weights(self, m):
        #_init_weights 方法用于初始化卷积层和线性层的权重和偏置。
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)#将权重初始化为截断正态分布，标准差为 0.02
            nn.init.constant_(m.bias, 0)# 将偏置初始化为 0

    def init_weights(self, pretrained=None):
        #方法用于初始化整个模型的权重，允许使用预训练的权重。
        # 没用上，疑似冗余
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.apply(_init_weights)

    def forward(self, x):
        outs = []
        x = self.stages[0](self.downsample_layers[0](x))#卷积后归一化：在提取特征后进行归一化，可以确保输出特征的分布稳定，减少后续层的计算负担。
        outs = [self.norm0(x)]
        x = self.stages[1](self.downsample_layers[1](x))#卷积前归一化：可以让卷积操作在更稳定的分布上进行，这对训练深层网络有利。
        outs.append(self.norm1(x))
        return outs

class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        activation = md.Activation(activation)
        super().__init__(conv2d, upsampling, activation)

class DecoderBlock(nn.Module):
    def __init__(self,cin,cadd,cout,):
        super().__init__()
        self.cin = (cin + cadd)
        self.cout = cout
        self.conv1 = md.Conv2dReLU(self.cin,self.cout,kernel_size=3,padding=1,use_batchnorm=True)
        self.conv2 = md.Conv2dReLU(self.cout,self.cout,kernel_size=3,padding=1,use_batchnorm=True)

    def forward(self, x1, x2=None):
        x1 = F.interpolate(x1, scale_factor=2.0, mode="nearest")
        if x2 is not None:
            x1 = torch.cat([x1, x2], dim=1)
        x1 = self.conv1(x1[:,:self.cin])
        x1 = self.conv2(x1)
        return x1

class ConvBNReLU(nn.Module):
    def __init__(self,in_c,out_c,ks,stride=1,norm=True,res=False):
        super(ConvBNReLU, self).__init__()
        if norm:
            self.conv = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=ks, padding = ks//2, stride=stride,bias=False),nn.BatchNorm2d(out_c),nn.ReLU(True))
        else:
            self.conv = nn.Conv2d(in_c, out_c, kernel_size=ks, padding = ks//2, stride=stride,bias=False)
        self.res = res
    def forward(self,x):
        if self.res:
            return (x + self.conv(x))
        else:
            return self.conv(x)

class FUSE1(nn.Module):
    def __init__(self,in_channels_list=(96,192,384,768)):
        super(FUSE1, self).__init__()
        self.c31 = ConvBNReLU(in_channels_list[2],in_channels_list[2],1)
        self.c32 = ConvBNReLU(in_channels_list[3],in_channels_list[2],1)
        self.c33 = ConvBNReLU(in_channels_list[2],in_channels_list[2],3)

        self.c21 = ConvBNReLU(in_channels_list[1],in_channels_list[1],1)
        self.c22 = ConvBNReLU(in_channels_list[2],in_channels_list[1],1)
        self.c23 = ConvBNReLU(in_channels_list[1],in_channels_list[1],3)

        self.c11 = ConvBNReLU(in_channels_list[0],in_channels_list[0],1)
        self.c12 = ConvBNReLU(in_channels_list[1],in_channels_list[0],1)
        self.c13 = ConvBNReLU(in_channels_list[0],in_channels_list[0],3)

    def forward(self,x):
        x,x1,x2,x3 = x
        h,w = x2.shape[-2:]
        x2 = self.c33(F.interpolate(self.c32(x3),size=(h,w))+self.c31(x2))
        h,w = x1.shape[-2:]
        x1 = self.c23(F.interpolate(self.c22(x2),size=(h,w))+self.c21(x1))
        h,w = x.shape[-2:]
        x = self.c13(F.interpolate(self.c12(x1),size=(h,w))+self.c11(x))
        return x,x1,x2,x3

class FUSE2(nn.Module):
    def __init__(self,in_channels_list=(96,192,384)):
        super(FUSE2, self).__init__()

        self.c21 = ConvBNReLU(in_channels_list[1],in_channels_list[1],1)
        self.c22 = ConvBNReLU(in_channels_list[2],in_channels_list[1],1)
        self.c23 = ConvBNReLU(in_channels_list[1],in_channels_list[1],3)

        self.c11 = ConvBNReLU(in_channels_list[0],in_channels_list[0],1)
        self.c12 = ConvBNReLU(in_channels_list[1],in_channels_list[0],1)
        self.c13 = ConvBNReLU(in_channels_list[0],in_channels_list[0],3)

    def forward(self,x):
        x,x1,x2 = x
        h,w = x1.shape[-2:]
        x1 = self.c23(F.interpolate(self.c22(x2),size=(h,w),mode='bilinear',align_corners=True)+self.c21(x1))
        h,w = x.shape[-2:]
        x = self.c13(F.interpolate(self.c12(x1),size=(h,w),mode='bilinear',align_corners=True)+self.c11(x))
        return x,x1,x2

class FUSE3(nn.Module):
    def __init__(self,in_channels_list=(96,192)):
        super(FUSE3, self).__init__()

        self.c11 = ConvBNReLU(in_channels_list[0],in_channels_list[0],1)
        self.c12 = ConvBNReLU(in_channels_list[1],in_channels_list[0],1)
        self.c13 = ConvBNReLU(in_channels_list[0],in_channels_list[0],3)

    def forward(self,x):
        x,x1 = x
        h,w = x.shape[-2:]
        x = self.c13(F.interpolate(self.c12(x1),size=(h,w),mode='bilinear',align_corners=True)+self.c11(x))
        return x,x1

class MID(nn.Module):
    def __init__(self, encoder_channels, decoder_channels):
        super().__init__()
        encoder_channels = encoder_channels[1:][::-1]#跳过第一个通道数并反转列表。[::n]为slice step
        self.in_channels = [encoder_channels[0]] + list(decoder_channels[:-1])
        self.add_channels = list(encoder_channels[1:]) + [96] #每一层添加的通道数。
        self.out_channels = decoder_channels
        self.fuse1 = FUSE1()
        self.fuse2 = FUSE2()
        self.fuse3 = FUSE3()
        decoder_convs = {}#创建一个空字典，用于存储解码器层。
        for layer_idx in range(len(self.in_channels) - 1):#外层循环遍历每一层的索引，除最后一层外的所有层。
            for depth_idx in range(layer_idx + 1):
                if depth_idx == 0:
                    in_ch = self.in_channels[layer_idx]
                    skip_ch = self.add_channels[layer_idx] * (layer_idx + 1)
                    out_ch = self.out_channels[layer_idx]
                else:
                    out_ch = self.add_channels[layer_idx]
                    skip_ch = self.add_channels[layer_idx] * (layer_idx + 1 - depth_idx)
                    in_ch = self.add_channels[layer_idx - 1]
                decoder_convs[f"x_{depth_idx}_{layer_idx}"] = DecoderBlock(in_ch, skip_ch, out_ch)
        decoder_convs[f"x_{0}_{len(self.in_channels)-1}"] = DecoderBlock(self.in_channels[-1], 0, self.out_channels[-1])
        self.decoder_convs = nn.ModuleDict(decoder_convs)

    def forward(self, *features):
        decoder_features = {}
        features = self.fuse1(features)[::-1]
        decoder_features["x_0_0"] = self.decoder_convs["x_0_0"](features[0],features[1])
        decoder_features["x_1_1"] = self.decoder_convs["x_1_1"](features[1],features[2])
        decoder_features["x_2_2"] = self.decoder_convs["x_2_2"](features[2],features[3])
        decoder_features["x_2_2"], decoder_features["x_1_1"], decoder_features["x_0_0"] = self.fuse2((decoder_features["x_2_2"], decoder_features["x_1_1"], decoder_features["x_0_0"]))
        decoder_features["x_0_1"] = self.decoder_convs["x_0_1"](decoder_features["x_0_0"], torch.cat((decoder_features["x_1_1"], features[2]),1))
        decoder_features["x_1_2"] = self.decoder_convs["x_1_2"](decoder_features["x_1_1"], torch.cat((decoder_features["x_2_2"], features[3]),1))
        decoder_features["x_1_2"], decoder_features["x_0_1"] = self.fuse3((decoder_features["x_1_2"], decoder_features["x_0_1"]))
        decoder_features["x_0_2"] = self.decoder_convs["x_0_2"](decoder_features["x_0_1"], torch.cat((decoder_features["x_1_2"], decoder_features["x_2_2"], features[3]),1))
        return self.decoder_convs["x_0_3"](torch.cat((decoder_features["x_0_2"], decoder_features["x_1_2"], decoder_features["x_2_2"]),1))


class DTD(SegmentationModel):
    def __init__(self, encoder_name = "resnet18", decoder_channels = (384, 192, 96, 64), classes = 1):
        super().__init__()
        self.vph = torch.load('/data/jinrong/wcw/PS/DocTamper/train_logs/pre_trained/vph_imagenet.pt')
        self.swin = torch.load('/data/jinrong/wcw/PS/DocTamper_hehe/checkp/swin_imagenet.pt')
        self.fph = FPH()
        self.decoder = MID(encoder_channels=(96, 192, 384, 768), decoder_channels=decoder_channels)
        self.segmentation_head = SegmentationHead(in_channels=decoder_channels[-1], out_channels=classes, upsampling=2.0)
        self.addcoords = AddCoords()
        self.FU = nn.Sequential(SCSEModule(448),nn.Conv2d(448,192,3,1,1),nn.BatchNorm2d(192),nn.ReLU(True))
        self.classification_head = None
        self.initialize()
#2,2,18,2
    def forward(self,x,dct,qt):
        #features0为vph输出，features1为vph+fph输出，features2、3为swins输出
        features = self.vph(self.addcoords(x))#(B,92,128,128)4x
        print(dct.shape,qt.shape)
        features[1] = self.FU(torch.cat((features[1],self.fph(dct,qt)),1))#输入为8x
        rst = self.swin[0](features[1].flatten(2).transpose(1,2).contiguous())#输入为(B,384,64,64),8x,输出为16x
        N,L,C = rst.shape
        H = W = int(L**(1/2))
        features.append(self.vph.norm2(rst.transpose(1,2).contiguous().view(N,C,H,W)))#16x
        features.append(self.vph.norm3(self.swin[2](self.swin[1](rst)).transpose(1,2).contiguous().view(N,C*2,H//2,W//2)))#输入为16x,输出为32x
        decoder_output = self.decoder(*features)
        return self.segmentation_head(decoder_output)

class seg_dtd(nn.Module):
    def __init__(self, model_name='resnet18', n_class=1):
        super().__init__()
        self.model = DTD(encoder_name=model_name, classes=n_class)

    @autocast()
    def forward(self, x, dct, qt):
        x = self.model(x, dct, qt)
        return x

