import os
import shutil
from PIL import Image

def create_folder_for_images(input_folder, output_folder):
    """
    为每张图像单独创建一个文件夹，并将图像存储到该文件夹中。
    
    :param input_folder: 输入的图像文件夹路径
    :param output_folder: 输出的目标文件夹路径
    """
    # 如果输出文件夹不存在，则创建
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有图像文件（假设为png或jpg格式）
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.png') or file_name.endswith('.jpg'):  # 根据需要可以添加其他格式
            image_path = os.path.join(input_folder, file_name)
            image = Image.open(image_path).convert('RGB')  # 打开图像并转换为RGB格式

            # 为每张图像创建一个新的文件夹，文件夹名称为图像的文件名（不带扩展名）
            image_name_without_extension = os.path.splitext(file_name)[0]
            image_folder_path = os.path.join(output_folder, image_name_without_extension)
            
            # 创建文件夹
            os.makedirs(image_folder_path, exist_ok=True)
            
            # 保存图像到新的文件夹中
            output_image_path = os.path.join(image_folder_path, file_name)
            image.save(output_image_path)
            
            print(f"Saved {file_name} to {image_folder_path}")

# 示例用法
input_folder = '/root/autodl-tmp/CLIC/professional'  # 输入图像文件夹路径
output_folder = '/root/autodl-tmp/clic_save/origin'  # 输出文件夹路径

create_folder_for_images(input_folder, output_folder)
