import os
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import monai
from monai.data import CacheDataset
from torch.cuda.amp import autocast


from dataset.cpg0000 import CPG0000
from models.models import MyStableDiffusion, MyInceptionV3, MyOpenPhenom

def main_worker(args):
    device = torch.device("cuda")

    path_csv_file = os.path.join(args.path_dataset, "csv", args.plate + ".csv")
    print(path_csv_file)

    cpg0000 = CPG0000(path_csv=path_csv_file)

    dataset = CacheDataset(cpg0000.csv_ds, transform=cpg0000.val_transforms, cache_rate=0, num_workers=args.num_workers, copy_cache=False)
    
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, persistent_workers=True, drop_last=False)

    my_sd_vae = MyStableDiffusion(device, args)
    my_op = MyOpenPhenom(device, args)
    my_inception = MyInceptionV3(device, args)

    ssim = monai.losses.SSIMLoss(spatial_dims=2, reduction="none")

    dict_metrics = {
        "kl": [],
        "mae": [],
        "ssim": [],
        "emd": [],
        "real_inception": [],
        "fake_inception": [],
        "real_openphenom": [],
        "fake_openphenom": []
    }

    for batch in loader:
        with torch.no_grad():
            with autocast():
                images = batch["IMAGE"].to(device)
                batch_size, patches, channels, height, width = images.shape

                images_rgb = torch.cat([images[:, :, :3, :, :], images[:, :, 2:, :, :]], dim=2).view(-1, 3, height, width) #(B*crops*2, 3, H, W)
                images = images.view(-1, channels, height, width)

                #KL and reconstruction
                latents, z_mu, z_logvar = my_sd_vae.encode_img(images_rgb)
                
                kl_loss = 0.5 * torch.sum(z_mu.pow(2) + torch.exp(z_logvar) - z_logvar - 1, dim=[1, 2, 3])

                dict_metrics["kl"].append(kl_loss.cpu().as_tensor().view(batch_size, patches, 2))

                reconstructed_rgb = my_sd_vae.decode_img(latents)
                reconstructed_temp = reconstructed_rgb.view(batch_size, patches, 2, 3, height, width)
                reconstructed = torch.cat([
                    reconstructed_temp[:, :, 0, :3],
                    reconstructed_temp[:, :, 1, 1:3]
                ], dim=2).view(-1, channels, height, width)
            
                #SSIM/L1
                ssim_channels = ssim(images.view(-1, 1, height, width), reconstructed.view(-1, 1, height, width)).view(-1, channels)
                dict_metrics["ssim"].append(ssim_channels.cpu().view(batch_size, patches, channels)) #Already a PyTorch Tensor

                loss = F.l1_loss(images.float(), reconstructed.float(), reduction="none")
                dict_metrics["mae"].append(torch.mean(loss, dim=(-1, -2)).cpu().as_tensor().view(batch_size, patches, channels))

                #EMD
                images_sorted = torch.sort(images.view(batch_size*patches, channels, -1), dim=-1).values
                reconstructed_sorted = torch.sort(reconstructed.view(batch_size*patches, channels, -1), dim=-1).values
                wass_dist = torch.mean(torch.abs(images_sorted - reconstructed_sorted), dim=-1)

                dict_metrics["emd"].append(wass_dist.cpu().as_tensor().view(batch_size, patches, channels))

                # Fork parallel tasks
                fut_real_inception = torch.jit.fork(my_inception.get_features, images_rgb)
                fut_fake_inception = torch.jit.fork(my_inception.get_features, reconstructed_rgb)
                fut_real_openphenom = torch.jit.fork(my_op.get_features, images)
                fut_fake_openphenom = torch.jit.fork(my_op.get_features, reconstructed)

                # Wait and collect results
                real_inception = torch.jit.wait(fut_real_inception)
                fake_inception = torch.jit.wait(fut_fake_inception)
                real_openphenom = torch.jit.wait(fut_real_openphenom)
                fake_openphenom = torch.jit.wait(fut_fake_openphenom)
                
                #Inception RGB
                dict_metrics["real_inception"].append(real_inception.cpu().as_tensor().view(batch_size, patches, 2, -1))
                dict_metrics["fake_inception"].append(fake_inception.cpu().as_tensor().view(batch_size, patches, 2, -1))

                #OpenPhenom
                dict_metrics["real_openphenom"].append(real_openphenom.cpu().as_tensor().view(batch_size, patches, channels, -1))
                dict_metrics["fake_openphenom"].append(fake_openphenom.cpu().as_tensor().view(batch_size, patches, channels, -1))

    
    for key, value in dict_metrics.items():
        folder_path = os.path.join(args.folder_save, key)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        save_path = os.path.join(args.folder_save, key, args.plate + ".pt")
        torch.save(torch.cat(value), save_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_dataset', default='/folder1/folder2/cpg0000-jump-pilot', type=str)
    parser.add_argument('--plate', default="BR00116991", type=str)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--folder_save', default='/folder1/folder2/cpg0000-jump-pilot/metrics', type=str)
    parser.add_argument('--cache_dir_hf', default='/folder1/folder2/cache_dir_hf', type=str)
    parser.add_argument('--torch_home', default='/folder1/folder2/torch_home', type=str)

    args = parser.parse_args()
    
    # Spawn processes
    main_worker(args=args)

if __name__=="__main__":
    main()