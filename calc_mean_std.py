import os
from PIL import Image
import torch
from torchvision import transforms
import numpy as np

# 设置文件夹路径
image_folder = '/root/autodl-tmp/CLIC/professional'

# 定义图像的转换操作
transform = transforms.Compose([
    transforms.ToTensor()  # 转换为Tensor，范围从0到1
])

# 初始化存储图像的均值和方差的变量
mean = torch.zeros(3)  # 三个通道(R, G, B)
std = torch.zeros(3)
n_samples = 0  # 图像计数

# 遍历文件夹中的所有图像
for filename in os.listdir(image_folder):
    if filename.endswith(('.jpg', '.png', '.jpeg')):  # 只处理图像文件
        image_path = os.path.join(image_folder, filename)
        
        # 打开图像
        image = Image.open(image_path).convert('RGB')
        
        # 对图像进行转换
        image_tensor = transform(image)
        
        # 计算每个图像的均值和方差
        mean += image_tensor.mean(dim=(1, 2))  # 按照通道计算均值
        std += image_tensor.std(dim=(1, 2))    # 按照通道计算方差
        n_samples += 1  # 图像计数

# 计算全局均值和方差
mean /= n_samples
std /= n_samples

print(f'全局均值: {mean}')
print(f'全局方差: {std}')
