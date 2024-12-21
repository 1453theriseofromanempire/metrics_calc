import os
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import numpy as np
from tqdm import tqdm

def pad_image(image, patch_size=256):
    """
    如果图像的高度或宽度不是256的倍数,使用零填充到最近的256的倍数。
    """
    c, h, w = image.shape  # 获取图像的通道数、宽度和高度
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    
    # 使用零填充图像
    padding = (0, pad_w, 0, pad_h)  # (左、上、右、下)
    padded_image = torch.nn.functional.pad(image, padding, mode='constant', value=0)
    return padded_image

def split_into_patches(image, patch_size=256, shift=False):
    """
    将图像划分为多个256x256的块。
    """
    c, h, w = image.shape  # 获取图像的维度
    patches = []

    patches_num = 0
    if not shift:
        patch_num_w = w // patch_size
        patch_num_h = h // patch_size
        for i in range(0, patch_num_h):
            for j in range(0, patch_num_w):
                idx_i = i * patch_size
                idx_j = j * patch_size
                patch = image[:, idx_i:idx_i+patch_size, idx_j:idx_j+patch_size]
                patches.append(patch)
                patches_num += 1
    else:
        patch_num_w = (w - (patch_size//2)) // patch_size
        patch_num_h = (h - (patch_size//2)) // patch_size
        for i in range(0, patch_num_h,):
            for j in range(0, patch_num_w):
                idx_i = i * patch_size + patch_size // 2
                idx_j = j * patch_size + patch_size // 2
                patch = image[:, idx_i:idx_i+patch_size, idx_j:idx_j+patch_size]
                patches.append(patch)
                patches_num += 1
    
    return patches, patches_num

def process_images_from_folder(image_folder, output_folder, shift_output_folder, patch_size=256):
    """
    从文件夹读取图像，分块后保存每个块到与原图文件名同名的子文件夹。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    total_patches_num = 0
    image_output_folder = "/root/autodl-tmp/DIV2K/real"
    # 获取文件夹中所有图像文件
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    cnt = 0
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_folder, image_file)
        image = Image.open(image_path).convert("RGB")  # 打开图像并转换为RGB格式
        image_tensor = transforms.ToTensor()(image)  # 转换为Tensor
        
        # 填充图像
        # padded_image = pad_image(image_tensor, patch_size=patch_size)
        image_name_without_ext = os.path.splitext(image_file)[0]
        # 切分图像
        patches, patches_num = split_into_patches(image_tensor, patch_size)
        total_patches_num += patches_num
        # 为每个图像创建一个对应的子文件夹
        # image_name_without_ext = os.path.splitext(image_file)[0]
        # image_output_folder = os.path.join(output_folder, image_name_without_ext)

        # if not os.path.exists(image_output_folder):
        #     os.makedirs(image_output_folder)

        # 按照文件名保存每个块到对应的子文件夹
        for idx, patch in enumerate(patches):
            patch_filename = f"{cnt}.png"  # 按照索引命名图像块
            patch_path = os.path.join(image_output_folder, patch_filename)
            save_image(patch, patch_path)  # 保存每个图像块
            cnt += 1
        

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        shift_patches, shift_patches_num = split_into_patches(image_tensor, patch_size, shift=True)
        total_patches_num += shift_patches_num
        # 为每个图像创建一个对应的子文件夹
        # image_name_without_ext = os.path.splitext(image_file)[0]
        # image_output_folder = os.path.join(shift_output_folder, image_name_without_ext)

        # if not os.path.exists(image_output_folder):
        #     os.makedirs(image_output_folder)

        # 按照文件名保存每个块到对应的子文件夹
        for idx, patch in enumerate(shift_patches):
            patch_filename = f"{cnt}.png"  # 按照索引命名图像块
            patch_path = os.path.join(image_output_folder, patch_filename)
            save_image(patch, patch_path)  # 保存每个图像块
            cnt += 1
    
    return total_patches_num
        

# 示例：处理图像文件夹
image_folder = "/root/autodl-tmp/DIV2K_valid_HR"  # 图像文件夹路径
output_folder = "/root/autodl-tmp/clic/real_img/not_shift"  # 输出图像块文件夹路径
shift_output_folder = "/root/autodl-tmp/clic/real_img/shift"

patches_num = process_images_from_folder(image_folder, output_folder, shift_output_folder)

print(f"Patches num: {patches_num}!")
