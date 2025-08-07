import os
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import monai

from dataset.lsun import LSUNDatasetWrapper
from models.models import MyStableDiffusion, MyInceptionV3

def main_worker(args):
    device = torch.device("cuda")

    dataset = LSUNDatasetWrapper(classes=[args.class_lsun])

    partition_size = len(dataset) // args.num_partition
    subsets = [Subset(dataset, list(range(i * partition_size, (i + 1) * partition_size))) for i in range(args.num_partition)]
    
    loader = DataLoader(subsets[args.index_partition], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, persistent_workers=True, drop_last=False)

    my_sd_vae = MyStableDiffusion(device, args)
    my_inception = MyInceptionV3(device, args)

    ssim = monai.losses.SSIMLoss(spatial_dims=2, reduction="none")

    dict_metrics = {
        "kl": [],
        "mae": [],
        "ssim": [],
        "emd": [],
        "real_inception": [],
        "fake_inception": []
    }

    for images in loader:
        with torch.no_grad():
            images = images.to(device)
            batch_size, channels, height, width = images.shape

            #KL and reconstruction
            latents, z_mu, z_logvar = my_sd_vae.encode_img(images)
            
            kl_loss = 0.5 * torch.sum(z_mu.pow(2) + torch.exp(z_logvar) - z_logvar - 1, dim=[1, 2, 3])

            dict_metrics["kl"].append(kl_loss.cpu())

            reconstructed = my_sd_vae.decode_img(latents)
        
            #SSIM/L1
            ssim_channels = ssim(images.view(-1, 1, height, width), reconstructed.view(-1, 1, height, width)).view(-1, channels)
            dict_metrics["ssim"].append(ssim_channels.cpu()) #Already a PyTorch Tensor

            loss = F.l1_loss(images.float(), reconstructed.float(), reduction="none")
            dict_metrics["mae"].append(torch.mean(loss, dim=(-1, -2)).cpu())

            #EMD
            images_sorted = torch.sort(images.view(batch_size, channels, -1), dim=-1).values
            reconstructed_sorted = torch.sort(reconstructed.view(batch_size, channels, -1), dim=-1).values
            wass_dist = torch.mean(torch.abs(images_sorted - reconstructed_sorted), dim=-1)

            dict_metrics["emd"].append(wass_dist.cpu())
            
            #Inception RGB
            dict_metrics["real_inception"].append(my_inception.get_features(images).cpu())
            dict_metrics["fake_inception"].append(my_inception.get_features(reconstructed).cpu())

    for key, value in dict_metrics.items():
        folder_path = os.path.join(args.folder_save, key)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        save_path = os.path.join(args.folder_save, key, str(args.index_partition) + ".pt")
        torch.save(torch.cat(value), save_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_lsun', default='classroom_train', type=str) #church_outdoor_train
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--folder_save', default='/folder1/folder2/sd-vae-lsun-classroom', type=str)
    parser.add_argument('--cache_dir_hf', default='/folder1/folder2/cache_dir_hf', type=str)
    parser.add_argument('--torch_home', default='/folder1/folder2/torch_home', type=str)
    parser.add_argument('--num_partition', default=100, type=int)
    parser.add_argument('--index_partition', default=0, type=int)

    args = parser.parse_args()
    
    # Spawn processes
    main_worker(args=args)

if __name__=="__main__":
    main()