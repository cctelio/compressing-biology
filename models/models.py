import os
import sys
folder_path = '../OpenPhenom/'
sys.path.append(folder_path)

import torch
import torch.nn as nn

from torchvision.models import inception_v3
from OpenPhenom.huggingface_mae import MAEModel
from diffusers import AutoencoderKL

from torchvision import transforms

class MyOpenPhenom:
    def __init__(self, device, args):
        self.model = MAEModel.from_pretrained("recursionpharma/OpenPhenom", cache_dir=args.cache_dir_hf)
        self.model.eval()
        self.model.to(device)
        self.model.return_channelwise_embeddings = True

    def get_features(self, crops):
        with torch.no_grad():
            embeddings = self.model.predict(crops)  #crops of shape (*, *, 256, 256)
        return embeddings

class MyStableDiffusion:
    def __init__(self, device, args):
        self.model = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", cache_dir=args.cache_dir_hf)
        self.model.eval()
        self.model.to(device)

    def encode_img(self, input_img):
        # Single image -> single latent in a batch
        if len(input_img.shape)<4:
            input_img = input_img.unsqueeze(0)
        with torch.no_grad():
            latent = self.model.encode(input_img*2 - 1) # Note scaling to [-1, 1]
            sample = 0.18215 * latent.latent_dist.sample()
            mu = latent.latent_dist.mean
            logvar = latent.latent_dist.logvar
        return sample, mu, logvar

    def decode_img(self, latents):
        # bath of latents -> list of images
        latents = (1 / 0.18215) * latents
        with torch.no_grad():
            image = self.model.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1) #Back to [0,1]
        image = image.detach()
        return image

class MyInceptionV3:
    def __init__(self, device, args):
        os.environ['TORCH_HOME'] = args.torch_home
        self.model = inception_v3(weights="Inception_V3_Weights.DEFAULT", transform_input=False)
        self.model.fc = nn.Identity()  # Remove the final classification layer
        self.model.eval()
        self.model.to(device)

    def get_features(self, images):

        transform_inceptionv3 = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ConvertImageDtype(torch.float), # Ensure image is float
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            ])

        images = torch.stack([transform_inceptionv3(image) for image in images])
        # images of shape (B, 3, 299, 299)
        with torch.no_grad():
            features = self.model(images)
        return features