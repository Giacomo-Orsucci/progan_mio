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
G = Generator(nch=nch, ws=True, pn=True).to(device)


G = torch.load('/home/giacomo/Desktop/progan_pretrained_celeba.pth')

G.eval()

RevealNet = StegaStampDecoder( #decoder and parameters passing
        IMAGE_RESOLUTION, IMAGE_CHANNELS, fingerprint_size=FINGERPRINT_SIZE
    )
RevealNet.load_state_dict(torch.load(args.decoder_path))
RevealNet = RevealNet.to(device)
bitwise_accuracy = 0

NUM_IMG = 2



for i in range(NUM_IMG):
    #casual generation of latent vector
    z = hypersphere(torch.randn(1, 4 * 32, 1, 1, device=device))  #also with hypersphere, the bitwise accuracy is not high
    #z = torch.randn(1, nch * 8, 1, 1, device=device) 
    
    with torch.no_grad():  
        image = G(z, G.max_res)
    save_image(image, os.path.join('/home/giacomo/Desktop/prova', f'image_{i}.png'), padding=0)

    detected_fingerprints = RevealNet(image)

    #"True" if the element is > 0 and "False" otherwise
    detected_fingerprints = (detected_fingerprints > 0).long()
    fingerprint = (fingerprint > 0).long()

    detected_fingerprints.to(device)
    fingerprint.to(device)

    print(fingerprint)
    
    print(detected_fingerprints)
    
    bitwise_accuracy += (detected_fingerprints == fingerprint).float().mean(dim=1).sum().item()
    

bitwise_accuracy = bitwise_accuracy / NUM_IMG #calcola l'accuratezza generale

print(f"Bitwise accuracy on fingerprinted images: {bitwise_accuracy}")
    
print("Successfully terminated")