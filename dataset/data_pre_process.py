from PIL import Image
import os
import cv2
import numpy as np
name = '30AC03A0012402236323出院记录.jpg'
# pic1 = rf'D:\PS\za_真实样本\01PS项目\PS转存\ml01\JPEG\{name}'
pic1 = rf'D:\PS\za_真实样本\01PS项目\ps后\ml_01\{name}'
# pic2 = rf'D:\PS\za_真实样本\01PS项目\待PS\ml_01\{name}'
# pic1 = rf'D:\PS\za_真实样本\01PS项目\tmp\ml1.jpg'
pic2 = rf'D:\PS\za_真实样本\01PS项目\tmp\ml2.jpg'# 新转存的

def read_image_as_array(file_path):
    # 以二进制模式读取图像文件
    with open(file_path, 'rb') as file:
        binary_data = file.read()

    # 将二进制数据转换为 NumPy 数组
    byte_array = np.asarray(bytearray(binary_data), dtype=np.uint8)

    # 使用 OpenCV 解码 NumPy 数组中的图像数据
    image = cv2.imdecode(byte_array, cv2.IMREAD_COLOR)

    return image


img_0 = read_image_as_array(pic1)
img_1 = read_image_as_array(pic2)



def get_mask(img_0, img_1,threshold=0):
    '''两个图片对比处理产生mask图'''
    difference = cv2.absdiff(img_0, img_1)
    gray_difference = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)

    # 设置一个阈值，将差异图像二值化，生成mask图
    _, mask = cv2.threshold(gray_difference, threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return mask


# 生成mask图
mask = get_mask(img_0, img_1)
mask = Image.fromarray(mask)
#展示mask图
mask.show()


e = os.listdir(r'D:\PS\za_真实样本\01PS项目\ml_01\ml_01\11')
for i in e:
    i.replace('mask_','')
e