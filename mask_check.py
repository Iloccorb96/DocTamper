import cv2
from PIL import Image
import os
import numpy as np

#
# path0 = r'D:\PS\za_真实样本\01PS项目\PS转存\ml01'
# path1 = r'D:\PS\za_真实样本\01PS项目\ps后\ml01'
# path0_list = os.listdir(path0)
# path1_list = os.listdir(path1)
def read_image_as_array(file_path):
    # 以二进制模式读取图像文件
    with open(file_path, 'rb') as file:
        binary_data = file.read()

    # 将二进制数据转换为 NumPy 数组
    byte_array = np.asarray(bytearray(binary_data), dtype=np.uint8)

    # 使用 OpenCV 解码 NumPy 数组中的图像数据
    image = cv2.imdecode(byte_array, cv2.IMREAD_COLOR)

    return image


# img_0 = read_image_as_array(r'D:\PS\za_真实样本\01PS项目\PS转存\fy_invoice\JPEG\202310081100300603780079177268正规医疗发票_0011.jpg')
# img_0 = read_image_as_array(r'D:\PS\za_真实样本\01PS项目\方远问题图片转存后.jpg')
# img_0 = read_image_as_array(r'D:\PS\za_真实样本\01PS项目\202310081100300603780079177268正规医疗发票_0011.jpg')
img_0 = read_image_as_array(r'D:\PS\za_真实样本\01PS项目\方远问题图片在方远侧另存为.jpg')
img_1 = read_image_as_array(r'D:\PS\za_真实样本\01PS项目\202310081100300603780079177268正规医疗发票_0011_2.jpg')
# img_1 = read_image_as_array(r'D:\PS\za_真实样本\01PS项目\ps后\fy_invoice\202310081100300603780079177268正规医疗发票_0011.jpg')
# os.listdir(r'D:\PS\za_真实样本\01PS项目\PS转存\ml01\JPEG')

# 两侧原图无差异
# 本地PS和本地另存为后无差异
# 方远侧两次PS图之间的无差异
# 方远侧PS后和本地PS后图片有差异
# 方远侧另存为和本地PS后图片有差异
# 方远侧另存为和方远侧PS图片无差异

def get_mask(img_0,img_1):
    '''两个图片对比处理产生mask图
    '''
    difference = cv2.absdiff(img_0, img_1)
    gray_difference = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)

    # 设置一个阈值，将差异图像二值化，生成mask图
    _, mask = cv2.threshold(gray_difference, 0, 255, cv2.THRESH_BINARY)
    mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
    return mask
mask = get_mask(img_0,img_1)

#使用cv2展示图片
img = Image.fromarray(mask)
img.show()