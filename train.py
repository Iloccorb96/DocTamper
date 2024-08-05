from utils.trainer import train_net
from utils import trainer
from dataset.DocTamperDataset import DocTamperDataset
from dataset.DocTamperDataset import train_transform, val_transform
from models.model_auxiliary import *
from PIL import Image
import os
from datetime import datetime
import pandas as pd

Image.MAX_IMAGE_PIXELS = 1000000000000000



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
model_name = 'dtd_transNeXt'
model = seg_dtd()
# model.load_state_dict(torch.load('/data/jinrong/wcw/PS/DocTamper/checkp/dtd_doctamper.pth',map_location='cuda')['state_dict'])
model = model.cuda()
# model = torch.nn.DataParallel(model)
for name, module in model.named_modules():
    if isinstance(module, nn.GELU):
        exec('model.' + torchmodify(name) + '=nn.GELU()')

# 模型保存路径
now = datetime.now()
formatted_time = now.strftime("%Y-%m-%d-%H")
save_ckpt_dir = os.path.join('/data/jinrong/wcw/PS/DocTamper/train_logs', model_name, 'ckpt',formatted_time)
save_log_dir = os.path.join('/data/jinrong/wcw/PS/DocTamper/train_logs', model_name,'train_log')
if not os.path.exists(save_ckpt_dir):
    os.makedirs(save_ckpt_dir)
if not os.path.exists(save_log_dir):
    os.makedirs(save_log_dir)


# 参数设置
param = {}
# # 准备数据集
# # data_dir1 = "/data/jinrong/wcw/PS/data/0000PSsample/patch_dataset/tampered/"
# train_data_dir = [    
#     '/data/jinrong/wcw/PS/data/000patch/train',
# #     '/data/jinrong/wcw/PS/data/000patch/006train_filter/',
# #     '/data/jinrong/wcw/PS/data/000patch/006SCD_train_filter/',
# #     '/data/jinrong/wcw/PS/data/000patch/006FCD_train/',
# #     '/data/jinrong/wcw/PS/data/000patch/007patch_filter/'
# ]
# train_image_list = []
# train_label_list = []
# for train_data in train_data_dir:
#     train_image_list += sorted([os.path.join(train_data, "image/",file) for file in os.listdir(os.path.join(train_data, "image/")) if not file.startswith('.')])
#     train_label_list += sorted([os.path.join(train_data, "label/",file) for file in os.listdir(os.path.join(train_data, "label/")) if not file.startswith('.')])

# test_data_dir = [    
#     '/data/jinrong/wcw/PS/data/000patch/val',
# #     '/data/jinrong/wcw/PS/data/000patch/006test_filter/',
# #     '/data/jinrong/wcw/PS/data/000patch/007patch_test/',
# ]
# test_image_list = []
# test_label_list = []
# for test_data in test_data_dir:
#     test_image_list += sorted([os.path.join(test_data, "image/",file) for file in os.listdir(os.path.join(test_data, "image/")) if not file.startswith('.')])
#     test_label_list += sorted([os.path.join(test_data, "label/",file) for file in os.listdir(os.path.join(test_data, "label/")) if not file.startswith('.')])
df = pd.read_pickle('/data/jinrong/wcw/PS/data/datasetMetadata.pkl')

train_image_list = df[df['dataset']=='trueSample_train']['image_path'].to_list()
test_image_list = df[df['dataset']=='trueSample_test']['image_path'].to_list()
train_label_list = df[df['dataset']=='trueSample_train']['label_path'].to_list()
test_label_list = df[df['dataset']=='trueSample_test']['label_path'].to_list()

param['train_image_list'] = train_image_list
param['test_image_list'] = test_image_list
param['train_label_list'] = train_label_list
param['test_label_list'] = test_label_list

param['batch_size'] = 4      # 批大小
param['disp_inter'] = 1       # 显示间隔(epoch)
param['save_inter'] = 4       # 保存间隔(epoch)
param['iter_inter'] = 200     # 显示迭代间隔(batch)
param['min_inter'] = 10
param['model_name'] = model_name          # 模型名称
param['save_log_dir'] = save_log_dir      # 日志保存路径
param['save_ckpt_dir'] = save_ckpt_dir    # 权重保存路径

param['epochs'] = 102    
param['warm_up'] = 2
param['save_epoch']=[11,21,31,41,51,61,71,81]
# 加载权重路径（继续训练）
param['load_ckpt_dir'] = None #'/data/jinrong/wcw/PS/DocTamper/train_logs/dtd_transNeXt/ckpt/2024-07-14-16/checkpoint-latest.pth'

#
# 训练
train_net(param, model)




    

