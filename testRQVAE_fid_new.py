import argparse, os, sys, datetime, glob, importlib

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf
import yaml
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torchvision
from PIL import Image

from torch_fidelity import calculate_metrics
from pytorch_fid import fid_score
import lpips

sys.path.append("/root/home/codes/imp_trans/src/taming-transformers")
sys.path.append("/root/home/codes/imp_trans/src/rq-vae-transformer-main")

from rqvae.img_datasets import create_dataset
from rqvae.models import create_model
from rqvae.metrics import compute_fid

cfg = "src/rq-vae-transformer-main/configs/imagenet256/stage1/in256-rqvae-8x8x4.yaml"

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def read_cfg(path):
    config = OmegaConf.load(path)
    return config

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def calculate_lpips(img1, img2):
    lpips_metric = lpips.LPIPS(net='vgg').to('cuda')
    return lpips_metric(img1, img2).mean().item()

def calculate_fid(real_images, generated_images):
    metrics = calculate_metrics(
        input1=real_images, 
        input2=generated_images, 
        metrics=['fid',]
    )
    return metrics["fid"]

def calc_metrics(xs, x_hat):
    psnr = calculate_psnr(xs, x_hat)
    Lpips = calculate_lpips(xs, x_hat)
    # fid = calculate_fid(xs, x_hat)

    return psnr, Lpips

def save_images_to_dir(images, path):
    import os
    from torchvision.utils import save_image
    if not os.path.exists(path):
        os.makedirs(path)

    for i, image in enumerate(images):
        save_image(image, os.path.join(path, f"{i}.png"))

def load_image_patches(patch_folder, patch_size=256):
    """
    从文件夹中加载图像块，并按块的顺序存储到列表中。
    """
    patch_files = sorted([f for f in os.listdir(patch_folder) if f.endswith('.png')])
    patches = []

    for patch_file in patch_files:
        patch_path = os.path.join(patch_folder, patch_file)
        patch = Image.open(patch_path).convert("RGB")
        patch_tensor = transforms.ToTensor()(patch)  # 转换为Tensor
        patch_tensor = transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))(patch_tensor)
        patches.append(patch_tensor)

    return patches, patch_files

class CustomImageDataset(Dataset):
    def __init__(self, root_path, transform=None):
        self.image_files = [f for f in os.listdir(root_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        self.root_path = root_path
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_path, self.image_files[idx])
        image = Image.open(img_name).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image
    
def create_dataset(image_folder_path, is_eval=False, logger=None):
    # Define transformations for training and validation datasets
    # transforms_trn = create_transforms(config.dataset, split='train', is_eval=is_eval)
    transforms_ = [
                    # transforms.Resize(256),
                    # transforms.CenterCrop(256),
                    # transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ]
    transforms_val = transforms.Compose(transforms_)

    # Get the root directory from config
    # root = config.dataset.get('root', None)

    # # Modify to load images from a single folder
    # root = root if root else '/root/autodl-tmp/ILSVRC2012_train'

    # Only use one folder for images (e.g., 'train' or 'val' folder)
    # image_folder_path = f"{root}/data"  # Adjust this if you want to point to a different folder
    dataset = CustomImageDataset(root_path=image_folder_path, transform=transforms_val)

    return dataset # Returning the same dataset for both training and validation


@torch.no_grad()
def eval(model, config, root_path, generated_image_root_path):
    model.eval()

    inception_model = torchvision.models.inception_v3(pretrained=True).eval()

    subfolders = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]

    real_image_root_path = root_path
    # generated_image_root_path = "/root/autodl-tmp/clic/fake"

    fid_avg = 0
    total_val_len = 0

    psnr_avg = 0
    lpips_avg = 0
    
    dataset_val = create_dataset(real_image_root_path)

    loader_val = DataLoader(
        dataset_val, shuffle=False, pin_memory=True,
        batch_size=config.experiment.batch_size,
        num_workers=14,
    )

    val_len = len(loader_val)
    total_val_len += val_len
    pbar = tqdm(enumerate(loader_val), total=len(loader_val), leave=True)
    # patch_size = 256

    # patches, patch_files = load_image_patches(patch_folder, patch_size)
    images_real = []
    images_generated = []

    for it, inputs in pbar:
        model.zero_grad(set_to_none=True)

        xs = inputs.to('cuda')
        xs = xs * 2.0 - 1.0
        x_hat, _, _ = model(xs)

        x_hat = torch.clamp(x_hat, -1., 1.)
        xs = xs * 0.5 + 0.5
        x_hat = x_hat * 0.5 + 0.5
        
        # psnr, Lpips = calc_metrics(xs, x_hat)

        # psnr_avg += psnr
        # lpips_avg += Lpips

        # Append images for FID calculation (ensure batching works well here)
        # images_real.append(xs.cpu())
        images_generated.append(x_hat.cpu())

    # Save the images to temporary directories for FID calculation
    # real_images = torch.cat(images_real, dim=0)
    generated_images = torch.cat(images_generated, dim=0)

    # 保存真实图像和生成图像到临时文件夹（用于FID计算）
    # real_image_path = os.path.join(real_image_root_path, subfolder)
    generated_image_path = generated_image_root_path
    
    # 创建目录并保存图像
    # save_images_to_dir(real_images, real_image_path)  
    save_images_to_dir(generated_images, generated_image_path)

# # 使用pytorch-fid计算FID
    fid_value = fid_score.calculate_fid_given_paths(
        paths=[real_image_root_path, generated_image_path],
        batch_size=config.experiment.batch_size,
        device='cuda',
        dims=2048  # 通常使用Inception v3的2048维度
    )
    # print(generated_image_path)
    # print(real_image_path)
    # fid_value = fid_score.calculate_fid_given_paths([real_image_path, generated_image_path],
    #                                          inception_model,
    #                                          transform=transform)

    fid_avg = fid_value
    # # Compute the averages

    return {
        # "psnr": psnr_avg,
        # "lpips": lpips_avg,
        "fid": fid_avg,
        "total_num": total_val_len
    }


def main():

    config = read_cfg(cfg)

    model, model_ema = create_model(config.arch, False)
    checkpoint = torch.load('/root/autodl-tmp/imagenet_1.4B_rqvae_50e/stage1/model.pt')
    model.load_state_dict(checkpoint['state_dict'])
    model.to('cuda')
    model.eval()

    # dataset_trn, dataset_val = create_dataset(config)

    # root_path = "/root/autodl-tmp/kodak"
    # root_path = "/root/autodl-tmp/clic"

    # ############### non-shifted img process #############################
    # no_shift_path = "real_img/not_shift"
    # generated_no_shift_path = "generated_img/not_shift"
    real_img_path = "/root/autodl-tmp/DIV2K/real"
    fake_img_path = "/root/autodl-tmp/DIV2K/fake"

    # root_not_shift_path = os.path.join(root_path, no_shift_path)
    # generated_image_root_not_shift_path = os.path.join(root_path, generated_no_shift_path)

    output_not_shift = eval(model, config, real_img_path, fake_img_path)

    ############### shifted img process #############################
    # shift_path = "real_img/shift"
    # generated_shift_path = "generated_img/shift"

    # root_shift_path = os.path.join(root_path, shift_path)
    # generated_image_root_shift_path = os.path.join(root_path, generated_shift_path)
    # output_shift = eval(model, config, root_shift_path, generated_image_root_shift_path)

    ############### calc metrics ###########################################
    total_num = output_not_shift['total_num'] 
    # psnr = (output_not_shift['psnr'] + output_shift['psnr']) / total_num
    # lpips = (output_not_shift['lpips'] + output_shift['lpips']) / total_num
    fid = output_not_shift['fid']
    print(f"FID: {fid} | Patch number: {total_num}")

    
if __name__ == "__main__":
    main()


