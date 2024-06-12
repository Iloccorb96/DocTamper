import os
import cv2
import numpy as np
# 图片路径
name = 'fy01'
pic_path_0 = rf'D:\PS\01PSproject\after_ps\{name}'
pic_path_1 = rf'D:\PS\01PSproject\saved_pic\{name}'
mask_path = rf'D:\PS\01PSproject\mask\{name}'

def remove_ipy(pic_list):
    for pic in pic_list:
        if pic == '.ipynb_checkpoints':
            pic_list.remove(pic)
    return pic_list
# 获取目录中的图片文件列表

pic_list_0 = sorted(remove_ipy(os.listdir(pic_path_0)))
pic_list_1 = sorted(remove_ipy(os.listdir(pic_path_1)))

def read_image_as_array(file_path):
    # 以二进制模式读取图像文件
    with open(file_path, 'rb') as file:
        binary_data = file.read()

    # 将二进制数据转换为 NumPy 数组
    byte_array = np.asarray(bytearray(binary_data), dtype=np.uint8)

    # 使用 OpenCV 解码 NumPy 数组中的图像数据
    image = cv2.imdecode(byte_array, cv2.IMREAD_COLOR)

    return image

def get_mask(img_0, img_1,threshold=0):
    '''两个图片对比处理产生mask图'''
    difference = cv2.absdiff(img_0, img_1)
    gray_difference = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)

    # 设置一个阈值，将差异图像二值化，生成mask图
    _, mask = cv2.threshold(gray_difference, threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return mask

# 确保掩码图保存目录存在
if not os.path.exists(mask_path):
    os.makedirs(mask_path)


# 没全p完，挨个处理
import tqdm
# for file_0 in tqdm.tqdm(pic_list_0):
#     for file_1 in pic_list_1:
#         if file_0 == file_1:
#             img_path_0 = os.path.join(pic_path_0, file_0)
#             img_path_1 = os.path.join(pic_path_1, file_1)
#
#             # 读取图像
#             img_0 = cv2.imread(img_path_0)
#             img_1 = cv2.imread(img_path_1)
#
#             # 检查图像是否读取成功
#             if img_0 is None or img_1 is None:
#                 print(f"Failed to read one of the images: {file_0}, {file_1}")
#                 continue
#
#             # 生成mask图
#             mask = get_mask(img_0, img_1)
#
#             # 保存mask图
#             mask_filename = os.path.join(mask_path, f"mask_{file_0}").replace('.jpg','.png')
#             cv2.imwrite(mask_filename, mask)
#             print(f"Saved mask: {mask_filename}")

# 全p完的，处理每对图像

for file_0, file_1 in zip(pic_list_0, pic_list_1):
    img_path_0 = os.path.join(pic_path_0, file_0)
    img_path_1 = os.path.join(pic_path_1, file_1)

    # 读取图像
    img_0 = read_image_as_array(img_path_0)
    img_1 = read_image_as_array(img_path_1)

    # 检查图像是否读取成功
    if img_0 is None or img_1 is None:
        print(f"Failed to read one of the images: {file_0}, {file_1}")
        continue

    # 生成mask图
    mask = get_mask(img_0, img_1)

    # 保存mask图
    mask_filename = os.path.join(mask_path, f"mask_{file_0}").replace('.jpg','.png')
    cv2.imencode('.png', mask)[1].tofile(mask_filename)
    print(f"Saved mask: {mask_filename}")

'''filename = r'D:\PS\01PSproject\待PS\gp02\90AC0144012402018784住院病案首页.jpg'
img = read_image_as_array(filename)
#使用cv2将图像矩阵顺时针旋转90度
img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
cv2.imencode('.jpg', img)[1].tofile(filename)
'''