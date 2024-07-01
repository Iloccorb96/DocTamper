from utils import trainer
from dataset import dataset
from dataset import train_transform, val_transform
from models.dtd import *

Image.MAX_IMAGE_PIXELS = 1000000000000000

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
device = torch.device("cuda")

# 准备数据集
data_dir = "../data/train_val_split/"
train_imgs_dir = os.path.join(data_dir, "train_images/")
val_imgs_dir = os.path.join(data_dir, "val_images/")

train_labels_dir = os.path.join(data_dir, "train_labels/")
val_labels_dir = os.path.join(data_dir, "val_labels/")
train_data = dataset(train_imgs_dir, train_labels_dir, transform=train_transform)
valid_data = dataset(val_imgs_dir, val_labels_dir, transform=val_transform)
def torchmodify(name):
    #model.layer1.0.GELU替换为model.layer1._modules['0'].GELU
    a=name.split('.')
    for i,s in enumerate(a) :
        if s.isnumeric() :
            a[i]="_modules['"+s+"']"  #将0替换为_modules['0']
    return '.'.join(a)
# 模型初始化
model_name = 'seg_dtd'
model = seg_dtd()
model.load_state_dict(torch.load('/data/jinrong/wcw/PS/DocTamper/checkp/dtd_doctamper.pth',map_location='cuda')['state_dict'])
model = model.cuda()
model = torch.nn.DataParallel(model)
for name, module in model.named_modules():
    if isinstance(module, nn.GELU):
        exec('model.' + torchmodify(name) + '=nn.GELU()')

# 模型保存路径
save_ckpt_dir = os.path.join('./outputs/', model_name, 'ckpt')
save_log_dir = os.path.join('./outputs/', model_name)
if not os.path.exists(save_ckpt_dir):
    os.makedirs(save_ckpt_dir)
if not os.path.exists(save_log_dir):
    os.makedirs(save_log_dir)


# 参数设置
param = {}
param['train_imgs_dir'] = os.path.join(data_dir, "train_images/")
param['val_imgs_dir'] = os.path.join(data_dir, "val_images/")
param['train_labels_dir'] = os.path.join(data_dir, "train_labels/")
param['val_labels_dir'] = os.path.join(data_dir, "val_labels/")

param['epochs'] = 45          # 训练轮数，请和scheduler的策略对应，不然复现不出效果，对于t0=3,t_mut=2的scheduler来讲，44的时候会达到最优；3个epoch的时间段后重置，并且每隔2个epoch调整一次学习率
param['batch_size'] = 4       # 批大小
param['disp_inter'] = 1       # 显示间隔(epoch)
param['save_inter'] = 4       # 保存间隔(epoch)
param['iter_inter'] = 50     # 显示迭代间隔(batch)
param['min_inter'] = 10

param['model_name'] = model_name          # 模型名称
param['save_log_dir'] = save_log_dir      # 日志保存路径
param['save_ckpt_dir'] = save_ckpt_dir    # 权重保存路径
param['T0']=3  #cosine warmup的参数
param['save_epoch']={2:[5,13,29,61],3:[8,20,44,92]}
# 加载权重路径（继续训练）
param['load_ckpt_dir'] = None

#
# 训练
best_model, model = trainer(param, model, train_data, valid_data)
