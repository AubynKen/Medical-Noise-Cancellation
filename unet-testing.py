# code for measuring PNSR and SSIM of our unet model
# author: Igor Augusto de Oliveira

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm # progress bar
from torch.optim import Adam
import numpy as np
from skimage import metrics
from PIL import Image
import sys

# Custom modules
from unet import StentDataset, UNet, ImageAugmentation

# Loading the model
######################
model_save_path = "weights/unet-exp1-iter_24000.pt"
model = UNet(in_channels=1, out_channels=1)
if torch.cuda.is_available():
    model.load_state_dict(torch.load(model_save_path))
    model.to("cuda")
else: 
    model.load_state_dict(torch.load(model_save_path, map_location=torch.device("cpu")))
model.eval()  # Switching to evaluation mode

# Generating test data
######################
n_test_images = 1000

# custom image generator
image_generator = ImageAugmentation(base_image_path="data/dataset/base_png")

# generate test images
image_generator.generate_input_images(n_images=n_test_images,
                                      save_path="data/dataset/test",
                                      generate_input=True,
                                      print_progress=False)

batch_size = 2
test_dataset = StentDataset(input_path="data/dataset/test/x",
                            target_path="data/dataset/test/y")
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

PSNR_0 = 0
SSIM_0 = 0
PSNR = 0
SSIM = 0
delta = 94
for i, data in enumerate(tqdm(test_loader)):
    # Get inputs and targets
    inputs, targets = data
    # Calculate outputs
    outputs = model(inputs)
    # Preprocess output tensors and targets for scikit-image
    inputs = inputs.cpu().detach().numpy()[:, 0, delta:-delta, delta:-delta]
    outputs = outputs.cpu().detach().numpy()[:, 0, ...]
    targets = targets.cpu().detach().numpy()[:, 0, ...]
    # Calculate metrics
    for j in range(batch_size):
        input = inputs[j, ...]
        output = outputs[j, ...]
        target = targets[j, ...]
        PSNR_0 = PSNR_0 + metrics.peak_signal_noise_ratio(target, input)/n_test_images
        SSIM_0 = SSIM_0 + metrics.structural_similarity(target, input)/n_test_images
        PSNR = PSNR + metrics.peak_signal_noise_ratio(target, output)/n_test_images
        SSIM = SSIM + metrics.structural_similarity(target, output)/n_test_images

print()
print("Testing results:")
print(f"PSNR_0: {PSNR_0}")
print(f"SSIM_0: {SSIM_0}")
print(f"PSNR: {PSNR}")
print(f"SSIM: {SSIM}")