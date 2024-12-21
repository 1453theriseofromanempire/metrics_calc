import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import re


def load_image_patches_from_folder(folder_path):
    """
    从指定文件夹加载图像 patches, 并返回它们的列表。
    假设所有的图像文件是按顺序命名的，如 0.png, 1.png, 2.png 等。
    """
    patch_files = os.listdir(folder_path)
    patch_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))

    print(patch_files)
    patches = []
    for file in patch_files:
        if file.endswith('.png'):  # 确保是图像文件
            patch_path = os.path.join(folder_path, file)
            patch_image = Image.open(patch_path).convert('RGB')
            patch_tensor = transforms.ToTensor()(patch_image)  # 转换为 Tensor
            patches.append(patch_tensor)
    return patches

def reconstruct_image_from_patches(patches, orig_width, orig_height, patch_size=256):
    """
    从 patches 重建图像，去除 padding 部分。
    """
    # # 计算拼接后的图像的行数和列数
    # num_patches_h = orig_height // patch_size
    # num_patches_w = orig_width // patch_size

    pad_h = (patch_size - orig_height % patch_size) % patch_size
    pad_w = (patch_size - orig_width % patch_size) % patch_size
    
    print(f"pad_h: {pad_h} | pad_w: {pad_w}")
    H = pad_h + orig_height
    W = pad_w + orig_width

    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    # 创建一个空白的张量，来存放重建的图像
    reconstructed_image = torch.zeros((3, H, W))  # 假设图像是RGB（三个通道） 
    
    patch_index = 0
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            patch = patches[patch_index]
            # 将 patch 放置到原图像中的正确位置
            reconstructed_image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size] = patch
            patch_index += 1
    
    # print(f"padded_img: {reconstructed_image.shape}")
    # # 去除 padding 部分：裁剪掉多余的行和列
    # pad_h = orig_height % patch_size
    # pad_w = orig_width % patch_size

    # 如果存在填充，则进行裁剪
    if pad_h > 0 and pad_w > 0:
        reconstructed_image = reconstructed_image[:, :-pad_h, :-pad_w]
    elif pad_h > 0 and pad_w == 0:
        reconstructed_image = reconstructed_image[:, :-pad_h, :]
    elif pad_h == 0 and pad_w > 0:
        reconstructed_image = reconstructed_image[:, :, :-pad_w]

    return reconstructed_image

def save_reconstructed_image(image_tensor, save_path):
    """
    保存重建后的图像为文件。
    """
    image_numpy = image_tensor.permute(1, 2, 0).numpy()  # 转换为 HWC 格式
    image_numpy = np.clip(image_numpy * 255, 0, 255).astype(np.uint8)  # 处理为图片范围
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(save_path)

# 示例：假设我们要从指定文件夹重建图像
root_path = '/root/autodl-tmp/clic_save/real_images'  # 存储 patch 的文件夹路径
output_image_root_path = '/root/autodl-tmp/clic_save/rec'  # 输出图像的保存路径
origin_img_root_path = "/root/autodl-tmp/CLIC/professional"

subfolders = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]

for subfolder in subfolders:
    # 1. 加载所有 patches
    folder_path = os.path.join(root_path, subfolder)
    patches = load_image_patches_from_folder(folder_path)

    origin_img_name = subfolder + ".png"
    origin_img_path = os.path.join(origin_img_root_path, origin_img_name)
    # print(origin_img_path)
    image = Image.open(origin_img_path)
    # 获取图像的尺寸
    origin_width, origin_height = image.size  # 返回的是宽度和高度
    print(f"origin_img: {origin_height}, {origin_width}")

    # 2. 重建原始图像
    reconstructed_image = reconstruct_image_from_patches(patches, origin_width, origin_height)
    # print(f"reconstructed_image: {reconstructed_image.shape}")
    # 3. 保存重建的图像
    recon_img_name = subfolder + "/" + subfolder + ".png"
    output_image_path = os.path.join(output_image_root_path, recon_img_name)
    save_reconstructed_image(reconstructed_image, output_image_path)

print(f"Reconstructed image saved to {output_image_path}")
