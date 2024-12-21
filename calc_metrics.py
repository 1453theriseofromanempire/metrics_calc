import os
import torch
from PIL import Image
from lpips import LPIPS
from skimage.metrics import peak_signal_noise_ratio as psnr
from pytorch_fid import fid_score
from torchvision import datasets, transforms

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 图像加载和转换
transform = transforms.Compose([
    # transforms.Resize((256, 256)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 根据你的数据调整
])

def load_images_from_folder(folder_path):
    """
    从指定文件夹加载所有图像，返回图像路径和图像列表。
    """
    image_names = []
    images = []
    for subfolder in sorted(os.listdir(folder_path)):
        subfolder_path = os.path.join(folder_path, subfolder)

        assert os.path.isdir(subfolder_path)

        for file_name in os.listdir(subfolder_path):
            if file_name.endswith('.png') or file_name.endswith('.jpg'):  # 根据需要支持不同格式
                image_path = os.path.join(subfolder_path, file_name)
                image = Image.open(image_path).convert('RGB')
                image = transform(image).unsqueeze(0)  # 添加批次维度
                img_name = os.path.splitext(file_name)[0]
                image_names.append(img_name)
                images.append(image)


    return image_names, images

def calculate_psnr(img1, img2):
    """
    计算 PSNR。
    """
    return psnr(img1.numpy().transpose(1, 2, 0), img2.numpy().transpose(1, 2, 0))  # 转换为HWC格式

@torch.no_grad()
def calculate_lpips(img1, img2, lpips_model):
    """
    计算 LPIPS。
    """
    return lpips_model.forward(img1, img2).item()

def calculate_fid(real_images_path, fake_images_path):
    """
    计算 FID。
    """
    # real_images = torch.cat(real_images, 0)
    # fake_images = torch.cat(fake_images, 0)
    
    fid_avg = 0
    idx = 0
    # for real_img, fake_img in zip(real_images, fake_images):   

    #     real_img = real_img.mul(255).byte()
    #     fake_img = fake_img.mul(255).byte()

    fid_value = fid_score.calculate_fid_given_paths(
        paths=[real_images_path, fake_images_path], 
        batch_size=1, 
        device=device, 
        dims=2048
    )
        # fid_avg += fid_value
        # idx += 1
    return fid_value

def main():
    real_folder = '/root/autodl-tmp/CLIC/professional'
    fake_folder = '/root/autodl-tmp/clic_save/recon'
    
    origin_path = '/root/autodl-tmp/clic_save/origin'
    rec_path = '/root/autodl-tmp/clic_save/rec'
    # 加载图像
    
    real_image_names, real_images = load_images_from_folder(origin_path)
    fake_image_names, fake_images = load_images_from_folder(rec_path)
    
    # 确保两个文件夹中的图像数量一致
    assert len(real_images) == len(fake_images), "两个文件夹中的图像数量不一致"
    
    # 初始化 LPIPS 模型
    lpips_model = LPIPS(net='vgg').to(device).eval()

    # 存储 PSNR、LPIPS 和 FID 结果
    psnr_total = 0
    lpips_total = 0
    fid_total = 0


    # 逐一计算 PSNR 和 LPIPS    
    for real_img_name, real_img, fake_img_name, fake_img in zip(real_image_names, real_images, fake_image_names, fake_images):
        assert real_img_name == fake_img_name

        psnr_val = calculate_psnr(real_img.squeeze(0), fake_img.squeeze(0))  # 删除批次维度
        lpips_val = calculate_lpips(real_img.to(device), fake_img.to(device), lpips_model)

        real_img_path = os.path.join(origin_path, real_img_name)
        rec_img_path = os.path.join(rec_path, fake_img_name)

        fid_val = calculate_fid(real_img_path, rec_img_path)

        fid_total += fid_val
        psnr_total += psnr_val
        lpips_total += lpips_val

        print(f"Image {real_img_name} fid: {fid_val} | psnr: {psnr_val} | lpips: {lpips_val}")

    # 计算平均 PSNR 和 LPIPS
    psnr_avg = psnr_total / len(real_images)
    lpips_avg = lpips_total / len(real_images)
    fid_avg = fid_total / len(real_images)

    print(f"Average PSNR: {psnr_avg}")
    print(f"Average LPIPS: {lpips_avg}")
    print(f"FID: {fid_avg}")

if __name__ == '__main__':
    main()
