from torch.cuda.amp import autocast, GradScaler#need pytorch>1.6
from models.fph import FPH
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from functools import partial
from segmentation_models_pytorch.base import modules as md
from segmentation_models_pytorch.base import SegmentationModel
from models.transnext2 import transnext_base
from models.auxiliary_modal import ModalitiesExtractor

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
    def __init__(self, dims=[96, 192,384,768], drop_path_rate=0.4, layer_scale_init_value=1e-6,multi=False):
        super().__init__()
        self.dims = dims
        self.multi = multi
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, 6)]#6个dropout率，等差数列
#         if self.multi:
#             self.downsample_layers = nn.ModuleList([nn.Sequential(nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
#                                                 LayerNorm(dims[0], eps=1e-6, data_format="channels_first")),
#                                        nn.Sequential(LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
#                                                 nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2))])
#         else:
        self.downsample_layers = nn.ModuleList([nn.Sequential(nn.Conv2d(6, dims[0], kernel_size=4, stride=4),
                                            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")),
                                   nn.Sequential(LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
                                            nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2))])
        self.stages = nn.ModuleList([nn.Sequential(*[ConvBlock(dim=dims[0], drop_path=dp_rates[j],
                                          layer_scale_init_value=layer_scale_init_value) for j in range(3)]), #layer_scale_init_value使得每层的输出的幅度变化更加平滑，避免梯度爆炸或梯度消失问题。
                                 nn.Sequential(*[ConvBlock(dim=dims[1], drop_path=dp_rates[3 + j],
                                                    layer_scale_init_value=layer_scale_init_value) for j in range(3)])])
    #stage:3个convblock
        self.apply(self._init_weights)#对模型中的每一层应用 _init_weights 方法，用于初始化每一层的权重。
        self.initnorm()#初始化4个layernorm层，可以通过类似 self.norm0 的方式进行访问。

    def initnorm(self):
        #初始化模型中的 LayerNorm层。
        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")## 创建LayerNorm层的实例，其中eps和data_format参数被固定。partial函数用于创建一个新的函数，该函数在调用时会带有预设的参数值。
        if self.multi:
            layer = norm_layer(self.dims[1])#部分函数实例化 LayerNorm 层，self.dims[i_layer] 作为 normalized_shape 传递。
            layer_name = f'norm{1}'# 构建LayerNorm层的名称，例如"norm0", "norm1", ..., "norm3"
            self.add_module(layer_name, layer)# add_module是Module类的一个方法，用于将子模块添加到模块中。
        else:
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
        
        x = self.stages[0](self.downsample_layers[0](x))#卷积后归一化：在提取特征后进行归一化，可以确保输出特征的分布稳定，减少后续层的计算负担。
        if not self.multi:
            outs = []
            outs = [self.norm0(x)]
            x = self.stages[1](self.downsample_layers[1](x))#卷积前归一化：可以让卷积操作在更稳定的分布上进行，这对训练深层网络有利。
            outs.append(self.norm1(x))
            return outs
        else:
            return self.stages[1](self.downsample_layers[1](x))
        

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
        x1 = F.interpolate(x1, scale_factor=2.0, mode="bilinear")#bilinear,nearest
        if x2 is not None:
            x1 = torch.cat([x1, x2], dim=1)
        x1 = self.conv1(x1[:,:self.cin])
        x1 = self.conv2(x1)
        return x1
    
# class Conv2dGroupNormReLU(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, groups=8):
#         super(Conv2dGroupNormReLU, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
#         self.norm = nn.GroupNorm(groups, out_channels)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         return self.relu(self.norm(self.conv(x)))
# class DecoderBlock(nn.Module):
#     '''
#     上采样两倍，x1和x2cat后，对x1前cin+cadd个通道进行两次3x3卷积，输出为cout
#     '''
#     def __init__(self,cin,cadd,cout,):
#         super().__init__()
#         self.cin = (cin + cadd)
#         self.cout = cout
#         self.conv1 = Conv2dGroupNormReLU(self.cin, self.cout,kernel_size=3, padding=1)
#         self.conv2 = Conv2dGroupNormReLU(self.cout,self.cout,kernel_size=3, padding=1)

#     def forward(self, x1, x2=None):
#         x1 = F.interpolate(x1, scale_factor=2.0, mode="bilinear")
#         if x2 is not None:
#             x1 = torch.cat([x1, x2], dim=1)
#         x1 = self.conv1(x1[:,:self.cin])
#         x1 = self.conv2(x1)
#         return x1

class ConvGNReLU(nn.Module):
    def __init__(self,in_c,out_c,ks,stride=1,norm=True,res=False):
        super(ConvGNReLU, self).__init__()
        if norm:
            self.conv = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=ks, padding=ks // 2, stride=stride, bias=False),
                nn.GroupNorm(num_groups=8, num_channels=out_c), nn.ReLU(True))
            # self.conv = nn.Sequential(
            # nn.Conv2d(in_c, out_c, kernel_size=ks, padding = ks//2, stride=stride,bias=False),
            # nn.BatchNorm2d(out_c),nn.ReLU(True))
        else:
            self.conv = nn.Conv2d(in_c, out_c, kernel_size=ks, padding = ks//2, stride=stride,bias=False)
        self.res = res
    def forward(self,x):
        if self.res:
            return (x + self.conv(x))
        else:
            return self.conv(x)

class FUSE1(nn.Module):
    '''从大到小输出跨尺度cat的4个层。逐层的，高层特征上采样到低层特征后尺度后，相加再卷积输出融合后的特征。
    '''
    def __init__(self,in_channels_list=(96,192,384,768)):
        super(FUSE1, self).__init__()
        self.c31 = ConvGNReLU(in_channels_list[2],in_channels_list[2],1)
        self.c32 = ConvGNReLU(in_channels_list[3],in_channels_list[2],1)
        self.c33 = ConvGNReLU(in_channels_list[2],in_channels_list[2],3)

        self.c21 = ConvGNReLU(in_channels_list[1],in_channels_list[1],1)
        self.c22 = ConvGNReLU(in_channels_list[2],in_channels_list[1],1)
        self.c23 = ConvGNReLU(in_channels_list[1],in_channels_list[1],3)

        self.c11 = ConvGNReLU(in_channels_list[0],in_channels_list[0],1)
        self.c12 = ConvGNReLU(in_channels_list[1],in_channels_list[0],1)
        self.c13 = ConvGNReLU(in_channels_list[0],in_channels_list[0],3)

    def forward(self,x):
        x,x1,x2,x3 = x
        h,w = x2.shape[-2:]
        x2 = self.c33(F.interpolate(self.c32(x3),size=(h,w))+self.c31(x2))#x3通道减半上采样到x2尺度经过3x3卷积后，和经过3x3卷积的x2相加,再3x3卷积
        h,w = x1.shape[-2:]
        x1 = self.c23(F.interpolate(self.c22(x2),size=(h,w))+self.c21(x1))
        h,w = x.shape[-2:]
        x = self.c13(F.interpolate(self.c12(x1),size=(h,w))+self.c11(x))
        return x,x1,x2,x3

class FUSE2(nn.Module):
    def __init__(self,in_channels_list=(96,192,384)):
        super(FUSE2, self).__init__()

        self.c21 = ConvGNReLU(in_channels_list[1],in_channels_list[1],1)
        self.c22 = ConvGNReLU(in_channels_list[2],in_channels_list[1],1)
        self.c23 = ConvGNReLU(in_channels_list[1],in_channels_list[1],3)

        self.c11 = ConvGNReLU(in_channels_list[0],in_channels_list[0],1)
        self.c12 = ConvGNReLU(in_channels_list[1],in_channels_list[0],1)
        self.c13 = ConvGNReLU(in_channels_list[0],in_channels_list[0],3)

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

        self.c11 = ConvGNReLU(in_channels_list[0],in_channels_list[0],1)
        self.c12 = ConvGNReLU(in_channels_list[1],in_channels_list[0],1)
        self.c13 = ConvGNReLU(in_channels_list[0],in_channels_list[0],3)

    def forward(self,x):
        x,x1 = x
        h,w = x.shape[-2:]
        x = self.c13(F.interpolate(self.c12(x1),size=(h,w),mode='bilinear',align_corners=True)+self.c11(x))
        return x,x1

class MID(nn.Module):
    def __init__(self, encoder_channels, decoder_channels):
        super().__init__()
        encoder_channels = encoder_channels[1:][::-1]#(768, 384, 192)
        self.in_channels = [encoder_channels[0]] + list(decoder_channels[:-1])#[768, 384, 192, 96]
        self.add_channels = list(encoder_channels[1:]) + [96] #每一层添加的通道数。[384, 192] -> [384, 192, 96]
        self.out_channels = decoder_channels#(384, 192, 96, 64)
        self.fuse1 = FUSE1()
        self.fuse2 = FUSE2()
        self.fuse3 = FUSE3()
        decoder_convs = {}#创建一个空字典，用于存储解码器层。
        for layer_idx in range(len(self.in_channels) - 1):#3层【0,1,2】layer的意思是特征图的层级，layer越大，特征图越大，layer2对应1/4
            for depth_idx in range(layer_idx + 1):#[0],[0,1],[0,1,2] depth的意思是该层往内的深度
                if depth_idx == 0:
                    in_ch = self.in_channels[layer_idx]
                    skip_ch = self.add_channels[layer_idx] * (layer_idx + 1)#乘几是因为cat了n个同层的特征图
                    out_ch = self.out_channels[layer_idx]
                else:
                    out_ch = self.add_channels[layer_idx]
                    skip_ch = self.add_channels[layer_idx] * (layer_idx + 1 - depth_idx)
                    in_ch = self.add_channels[layer_idx - 1]
                #DecoderBlock,上采样两倍，x1和x2cat后，对x1前cin+cadd个通道进行两次3x3卷积，输出为cout
                decoder_convs[f"x_{depth_idx}_{layer_idx}"] = DecoderBlock(in_ch, skip_ch, out_ch)
        decoder_convs[f"x_{0}_{len(self.in_channels)-1}"] = DecoderBlock(96*4, 0, self.out_channels[-1])
        self.decoder_convs = nn.ModuleDict(decoder_convs)

    def forward(self, *features):
        decoder_features = {}
        features = self.fuse1(features)[::-1]#倒序后，从小到大的融合的特征图，同层通道数不变
        #将新的features，从高到底，两两合并为一个，4层特征图变为3层特征图
        decoder_features["x_0_0"] = self.decoder_convs["x_0_0"](features[0],features[1])# Conv2d(1152, 384)
        decoder_features["x_1_1"] = self.decoder_convs["x_1_1"](features[1],features[2])# Conv2d(576, 192)
        decoder_features["x_2_2"] = self.decoder_convs["x_2_2"](features[2],features[3])# Conv2d(288, 96)
        #从小到大的融合的特征图，x_2_2:D01，x_1_1:D11，x_0_0:D21
        decoder_features["x_2_2"], decoder_features["x_1_1"], decoder_features["x_0_0"] = self.fuse2((decoder_features["x_2_2"], decoder_features["x_1_1"], decoder_features["x_0_0"]))
        #下面这行，将D10和D11cat后和D21融合生成D12
        decoder_features["x_0_1"] = self.decoder_convs["x_0_1"](decoder_features["x_0_0"], torch.cat((decoder_features["x_1_1"], features[2]),1))
        #下面这行，将D00和D01cat后和D11融合生成D02
        decoder_features["x_1_2"] = self.decoder_convs["x_1_2"](decoder_features["x_1_1"], torch.cat((decoder_features["x_2_2"], features[3]),1))
        #从小到大的融合的特征图，x_0_1:D12，x_1_2:D02
        decoder_features["x_1_2"], decoder_features["x_0_1"] = self.fuse3((decoder_features["x_1_2"], decoder_features["x_0_1"]))
        #x_0_2:D03
        decoder_features["x_0_2"] = self.decoder_convs["x_0_2"](decoder_features["x_0_1"], torch.cat((decoder_features["x_1_2"], decoder_features["x_2_2"], features[3]),1))#一个192和3个96
        #x_0_3:conv(96,64),相当于只用了x_0_2即D03，channel=96*3
        return self.decoder_convs["x_0_3"](torch.cat((decoder_features["x_0_2"], decoder_features["x_1_2"], decoder_features["x_2_2"],features[3]),1))


class DTD(SegmentationModel):
    def __init__(self, decoder_channels = (384, 192, 96, 64), classes = 2,modallity = ['SRM','Bayar','Noiseprint++']):
        super().__init__()
        self.vph = VPH()
        state_dict = torch.load('/data/jinrong/wcw/PS/DocTamper/train_logs/pre_trained/vph_imagenet2.pt')
        self.vph.load_state_dict(state_dict)
#         self.auxiliary_modal = ModalitiesExtractor(['noiseprint', 'bayar', 'srm'], '/data/jinrong/wcw/PS/DocTamper/train_logs/pre_trained/np++.pth')
#         self.len_aux = len(modallity)
#         for i in range(self.len_aux):
#             layer_name = f'vph{i}'
#             layer = VPH(multi=True)
#             state_dict_a = torch.load('/data/jinrong/wcw/PS/DocTamper/train_logs/pre_trained/auxiliary_vph_imagenet2.pt')
#             layer.load_state_dict(state_dict_a)
#             self.add_module(layer_name, layer)
        self.transnext = transnext_base()#参数待定
        state_dict = torch.load('/data/jinrong/wcw/PS/DocTamper/train_logs/pre_trained/transnext_base_backbone.pt')
        self.transnext.load_state_dict(state_dict)
        self.fph = FPH()
        self.decoder = MID(encoder_channels=(96, 192, 384, 768), decoder_channels=decoder_channels)
        self.segmentation_head = SegmentationHead(in_channels=decoder_channels[-1], out_channels=classes, upsampling=2.0)
        self.addcoords = AddCoords()
        # self.FU = nn.Sequential(SCSEModule(448),nn.Conv2d(448,192,3,1,1),nn.BatchNorm2d(192),nn.ReLU(True))
#         self.FU = nn.Sequential(SCSEModule(448+self.len_aux*192), nn.Conv2d(448+self.len_aux*192, 192, 3, 1, 1),
#                                 nn.GroupNorm(num_groups=8, num_channels=192), nn.ReLU(True))
        self.FU = nn.Sequential(SCSEModule(448), nn.Conv2d(448, 192, 3, 1, 1),
                                nn.GroupNorm(num_groups=8, num_channels=192), nn.ReLU(True))
        self.classification_head = None
        self.initialize()
#         self._initialize()
        
        
#2,2,18,2
    def forward(self,x,dct,qt):
        #features0为vph输出，features1为vph+fph输出，features2、3为swins输出
        features = self.vph(self.addcoords(x))#4x,一个通道加上坐标即可
#         auxiliary_features = self.auxiliary_modal(x)
#         auxiliary_features = [getattr(self, f'vph{i}')(self.addcoords(feature)) for i, feature in enumerate(auxiliary_features)]
#         features[1] = self.FU(torch.cat([features[1],*auxiliary_features,self.fph(dct,qt)],1))#8x
        features[1] = self.FU(torch.cat([features[1], self.fph(dct,qt)],1))
        transnext_rst  = self.transnext((features[1].flatten(2).transpose(1,2).contiguous(),features[1].shape[2:]))#features[1]输入为(B,192,64,64)
#         print(transnext_rst[0].shape,transnext_rst[1].shape)
        features = [features[0],features[1],self.vph.norm2(transnext_rst[0].transpose(1,2).reshape([-1,384,32,32])),self.vph.norm3(transnext_rst[1])]
        decoder_output = self.decoder(*features)
        return self.segmentation_head(decoder_output)
    
#     def _initialize(self):
#         # 遍历模型的模块
#         for module in self.children():
#             # 避开已经加载预训练权重的模块
#             if isinstance(module, (VPH, transnext_base,MID,SegmentationHead)):
#                 continue
#             # 遍历模块内的所有参数
#             for m in module.modules():
#                 if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#                     init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                     if m.bias is not None:
#                         init.constant_(m.bias, 0)
#                 elif isinstance(m, nn.BatchNorm2d):
#                     init.constant_(m.weight, 1)
#                     init.constant_(m.bias, 0)
#                 elif isinstance(m, LayerNorm):
#                     init.constant_(m.weight, 1)
#                     init.constant_(m.bias, 0)

                
class seg_dtd(nn.Module):
    def __init__(self, n_class=2):
        super().__init__()
        self.model = DTD(classes=2)

    @autocast()
    def forward(self, x, dct, qt):
        x = self.model(x, dct, qt)
        return x