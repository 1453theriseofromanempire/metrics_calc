import argparse, os, sys, datetime, glob, importlib

from cv2 import HOUGH_MULTI_SCALE
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf
import yaml
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
from PIL import Image

from torch_fidelity import calculate_metrics
from pytorch_fid import fid_score
import lpips
from DISTS_pytorch import DISTS

sys.path.append("/root/home/codes/imp_trans/src/taming-transformers")
sys.path.append("/root/home/codes/imp_trans/src/rq-vae-transformer-main")

from rqvae.img_datasets import create_dataset
from rqvae.models import create_model

cfg = "src/rq-vae-transformer-main/configs/imagenet256/stage1/in256-rqvae-8x8x8.yaml"

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
    lpips_metric = lpips.LPIPS(net="alex").eval().to('cuda')
    return lpips_metric(img1, img2).item()

def calculate_dists(img1, img2):
    dists_metric = DISTS().eval().to('cuda')
    return dists_metric(img1, img2).item()

def calculate_fid(real_images, generated_images):
    metrics = calculate_metrics(
        input1=real_images, 
        input2=generated_images, 
        metrics=['fid',]
    )
    return metrics["fid"]

def calc_metrics(xs, x_hat):
    psnr = calculate_psnr(xs, x_hat)
    Lpips = calculate_lpips(xs * 2. - 1., x_hat * 2. - 1.)
    dists = calculate_dists(xs, x_hat)
    # fid = calculate_fid(xs, x_hat)

    return psnr, Lpips, dists

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
                    # transforms.Normalize([0.4751, 0.4513, 0.4333], [0.2443, 0.2358, 0.2421]),
                    # transforms.Normalize([0.4751, 0.4513, 0.4333], [0.4886, 0.4716, 0.4842]),
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

def inverse_transform(tensor, mean, std):
    # 恢复标准化
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)  # 反向操作: t = t * std + mean
    return tensor


@torch.no_grad()
def eval(model, config, root_path):
    model.eval()
    
    subfolders = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]

    real_image_root_path = root_path
    # generated_image_root_path = "/root/autodl-tmp/kodak/generated_img/not_shift"

    fid_avg = 0
    total_val_len = 0

    psnr_avg = 0
    lpips_avg = 0

    # for subfolder in subfolders:
    #     subfolder_path = os.path.join(root_path, subfolder)

    dataset_val = create_dataset(root_path)

    loader_val = DataLoader(
        dataset_val, shuffle=False, pin_memory=True,
        batch_size=1,
        num_workers=14,
    )

    val_len = len(loader_val)
    total_val_len += val_len
    pbar = tqdm(enumerate(loader_val), total=len(loader_val), leave=True)
# patch_size = 256

# patches, patch_files = load_image_patches(patch_folder, patch_size)
    images_real = []
    images_generated = []

    # x_mean = torch.tensor([0.4751, 0.4513, 0.4333])
    # # x_std = torch.tensor([0.2443, 0.2358, 0.2421])
    # x_std = torch.tensor([0.4886, 0.4716, 0.4842])

    xs_mean = torch.zeros(3) 
    xs_std = torch.zeros(3)

    psnr_list = []
    lpips_list = []
    dists_list = []

    for it, inputs in pbar:
        model.zero_grad(set_to_none=True)

        xs = inputs.to('cuda')
        # xs_mean += xs.mean(dim=(1, 2))
        # xs_std += xs.std(dim=(1, 2))

        
        # xs_mean += xs.mean(dim=(0, 2, 3)).cpu()
        # xs_std += xs.std(dim=(0, 2, 3)).cpu()
        xs = xs * 2.0 - 1.0
        _, _, h, w = xs.shape
        patch_size = 256
        # h_scale = (h // 4) * 4
        # w_scale = (w // 4) * 4
        h_pad = (patch_size - h % patch_size) % patch_size
        w_pad = (patch_size - w % patch_size) % patch_size
        padding = (0, w_pad, 0, h_pad)
        padded_image = torch.nn.functional.pad(xs, padding, mode='constant', value=0)
        # xs = transforms.transforms.CenterCrop((h_scale, w_scale))(xs)


        # print(xs)

        
        x_hat, _, _ = model(padded_image)
        x_hat = x_hat[:, :, 0:h, 0:w]
        # print(x_hat)

        x_hat = torch.clamp(x_hat, -1., 1.)
        xs = xs * 0.5 + 0.5
        x_hat = x_hat * 0.5 + 0.5

        # xs = xs.mul_(x_mean).add_(x_std)
        # x_hat = x_hat.mul_(x_mean).add_(x_std)

        # xs = inverse_transform(xs, x_mean, x_std)
        # x_hat = inverse_transform(x_hat, x_mean, x_std)
        
        # print(xs.shape)
        # print(x_hat.shape)
        psnr, Lpips, dists = calc_metrics(xs, x_hat)

        psnr_list.append(psnr)
        lpips_list.append(Lpips)
        dists_list.append(dists)

        # Append images for FID calculation (ensure batching works well here)
        # images_real.append(xs.cpu())
        images_generated.append(x_hat.cpu())

    #     # Save the images to temporary directories for FID calculation
        # real_images = torch.cat(images_real, dim=0)
    # generated_images = torch.cat(images_generated, dim=0)

#     # 保存真实图像和生成图像到临时文件夹（用于FID计算）
    # real_image_path = os.path.join(real_image_root_path, subfolder)
    # generated_image_path = os.path.join(generated_image_root_path, subfolder)
    
#     # 创建目录并保存图像
    # save_images_to_dir(real_images, real_image_path)  
    save_images_to_dir(images_generated, "/root/home/codes/imp_trans/fake_img")

    # # # 使用pytorch-fid计算FID
    #     fid_value = fid_score.calculate_fid_given_paths(
    #         paths=[real_image_path, generated_image_path],
    #         batch_size=config.experiment.batch_size,
    #         device='cuda',
    #         dims=2048  # 通常使用Inception v3的2048维度
    #     )
    #     fid_avg += fid_value * len(images_generated)
    # # Compute the averages

    psnr_avg = sum(psnr_list)
    lpips_avg = sum(lpips_list)
    dists_avg = sum(dists_list)

    return {
        "psnr": psnr_avg,
        "lpips": lpips_avg,
        "dists": dists_avg,
        'mean': xs_mean,
        'std': xs_std,
        # "fid": fid_avg,
        "total_num": total_val_len
    }


def main():

    config = read_cfg(cfg)

    model, model_ema = create_model(config.arch, False)
    checkpoint = torch.load('/root/autodl-tmp/imagenet_1.4B_rqvae_50e/stage1/model.pt')
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.to('cuda')
    model.eval()

    # dataset_trn, dataset_val = create_dataset(config)

    # root_path = "/root/autodl-tmp/kodak"
    root_path = "/root/autodl-tmp/DIV2K_valid_HR"

    # ############### non-shifted img process #############################
    # no_shift_path = "real_img/not_shift"
    # generated_no_shift_path = "generated_img/not_shift"

    # root_not_shift_path = os.path.join(root_path, no_shift_path)
    # generated_image_root_not_shift_path = os.path.join(root_path, generated_no_shift_path)

    output_not_shift = eval(model, config, root_path)

    # ############### shifted img process #############################
    # shift_path = "real_img/shift"
    # generated_shift_path = "generated_img/shift"

    # root_shift_path = os.path.join(root_path, shift_path)
    # generated_image_root_shift_path = os.path.join(root_path, generated_shift_path)
    # output_shift = eval(model, config, root_shift_path, generated_image_root_shift_path)

    ############### calc metrics ###########################################
    total_num = output_not_shift['total_num']
    psnr = output_not_shift['psnr'] / total_num
    lpips = output_not_shift['lpips'] / total_num
    xs_mean = output_not_shift['mean'] / total_num
    xs_std = output_not_shift['std'] / total_num
    dists = output_not_shift['dists'] / total_num
    # fid = (output_not_shift['fid'] + output_shift['fid']) / total_num
    print(f"PSNR: {psnr} | LPIPS: {lpips} | DISTS: {dists}| mean: {xs_mean} | std: {xs_std}")

    
if __name__ == "__main__":
    main()


