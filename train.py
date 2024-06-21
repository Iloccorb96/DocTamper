from utils.trainer import train_net
from models.dtd import *
# from PIL import Image
# Image.MAX_IMAGE_PIXELS = 1000000000000000
import os
import sys
sys.path.append('/data/jinrong/wcw/PS/DocTamper/models')
from collections import OrderedDict
import os
from datetime import datetime
torch.backends.cudnn.enabled=False


# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
device = torch.device("cuda")



def torchmodify(name):
    #model.layer1.0.GELU替换为model.layer1._modules['0'].GELU
    a=name.split('.')
    for i,s in enumerate(a) :
        if s.isnumeric() :
            a[i]="_modules['"+s+"']"  #将0替换为_modules['0']
    return '.'.join(a)
# 模型初始化
model_name = 'seg_dtd'
model = seg_dtd('',2)
model = model.cuda()
for name, module in model.named_modules():
    if isinstance(module, nn.GELU):
        exec('model.' + torchmodify(name) + '=nn.GELU()')


#加载权重
state_dict = torch.load('/data/jinrong/wcw/PS/DocTamper_hehe/checkp/dtd_doctamper.pth')['state_dict']
## 单卡时不需要
# model = torch.nn.DataParallel(model)
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] if k.startswith('module.') else k  # remove `module.` 前缀
    new_state_dict[name] = v
state_dict = new_state_dict

model.load_state_dict(state_dict) 

# 模型保存路径
output_path = '/data/jinrong/wcw/PS/DocTamper/train_logs'
current_date = datetime.now().strftime('%Y%m%d')
save_ckpt_dir = os.path.join(output_path, 'ckpt', current_date)
save_log_dir = os.path.join(output_path,'train_log',current_date)
os.makedirs(save_ckpt_dir, exist_ok=True)
os.makedirs(save_log_dir, exist_ok=True)


# 参数设置
# 准备数据集
data_dir = '/data/jinrong/wcw/PS/data/0000PSsample/patch_dataset/tampered'
param = {}
param['train_imgs_dir'] = os.path.join(data_dir, "train_images/")
param['val_imgs_dir'] = os.path.join(data_dir, "val_images/")
param['train_labels_dir'] = os.path.join(data_dir, "train_labels/")
param['val_labels_dir'] = os.path.join(data_dir, "val_labels/")

# param['epochs'] = 93         # 训练轮数，请和scheduler的策略对应，不然复现不出效果，对于t0=3,t_mut=2的scheduler来讲，44的时候会达到最优；3个epoch的时间段后重置，并且每隔2个epoch调整一次学习率
param['epochs'] = 52          # 训练轮数=余弦周期+warmup
param['batch_size'] = 8      # 批大小
param['disp_inter'] = 1       # 显示间隔(epoch)
param['save_inter'] = 4       # 保存间隔(epoch)
param['iter_inter'] = 10     # 显示迭代间隔(batch)
param['min_inter'] = 10

param['model_name'] = model_name          # 模型名称
param['save_log_dir'] = save_log_dir      # 日志保存路径
param['save_ckpt_dir'] = save_ckpt_dir    # 权重保存路径
param['T0']=3  #cosine warmup的参数
param['T_max'] = 52
param['warm_up'] = 2
param['save_epoch']={2:[5,13,29,61],3:[8,20,44,92]}
# 加载权重路径（继续训练）
param['load_ckpt_dir'] = None

#
# 训练
best_model, model = train_net(param, model)
