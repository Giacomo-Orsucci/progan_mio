from model import Generator
from models import StegaStampDecoder
import torch

import torchvision.transforms as transforms
import torchvision
import numpy as np

import torch
from torchvision.utils import save_image
import os
import argparse
from utils import *

parser = argparse.ArgumentParser()

parser.add_argument(
    "--decoder_path",
    type=str,
    help="Provide trained StegaStamp decoder to verify fingerprint detection accuracy.",
)

parser.add_argument(
    "--image_resolution", type=int, help="Height and width of square images."
)

#Set the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

fingerprint = torch.tensor([0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,0,1,1,1,0,1,0,1,1,1,1,1,1,1,1,0,0,1,1,1,
                            0,1,0,0,0,0,0,1,1,1,1,1,0,1,1,0,1,0,1,0,1,1,0,0,0,0,0,0,0,0,1,1,0,1,1,1,1,
                            0,1,0,1,1,1,0,1,0,1,0,1,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0]).to(device) #embedded fingerprint



args = parser.parse_args()

IMAGE_RESOLUTION = args.image_resolution
IMAGE_CHANNELS = 3


FINGERPRINT_SIZE = len(fingerprint)


nch = 16
G = Generator().to(device)
G = torch.load('/media/giacomo/hdd_ubuntu/Progan_pretrained_celeba/ProGAN_150k_CelebA_128x128_generator.pth')

G.eval()

RevealNet = StegaStampDecoder( #decoder and parameters passing
        IMAGE_RESOLUTION, IMAGE_CHANNELS, fingerprint_size=FINGERPRINT_SIZE
    )
RevealNet.load_state_dict(torch.load(args.decoder_path))
RevealNet = RevealNet.to(device)
RevealNet.eval()
bitwise_accuracy = 0

NUM_IMG = 50000


for i in range(NUM_IMG):
    
    #casual generation of latent vector
    z = torch.randn(1, 128, 1, 1, device=device) # (Batch size, Channels, Height, Width)
    
    print(i) 
    image = G(z, G.max_res) #latent vector, 4*2^5 = 128 (max_res)
    save_image(image, os.path.join('/media/giacomo/hdd_ubuntu/progan_gen_50k', f'image_{i}.png'), padding=0)

    print(image)

    detected_fingerprints = RevealNet(image)

    #"True" if the element is > 0 and "False" otherwise
    detected_fingerprints = (detected_fingerprints > 0).long()
    fingerprint = (fingerprint > 0).long()

    detected_fingerprints.to(device)
    fingerprint.to(device)

    print(fingerprint)
    
    print(detected_fingerprints)
    
    bitwise_accuracy += (detected_fingerprints == fingerprint).float().mean(dim=1).sum().item()

    

bitwise_accuracy = bitwise_accuracy / (NUM_IMG) #compute the general accuracy

print(f"Bitwise accuracy on fingerprinted images: {bitwise_accuracy}")
    
print("Successfully terminated")