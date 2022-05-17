import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm # progress bar
from torch.optim import Adam
import numpy as np
from skimage import metrics

# Custom modules
from unet import StentDataset, UNet, ImageAugmentation

# Loading the model
######################
model_save_path = "/home/igor/Work/CS/PP22/Medical-Noise-Cancellation/saved_models/unet-exp1-iter_24000.pt"
model = UNet(in_channels=1, out_channels=1)
model.load_state_dict(torch.load(model_save_path, map_location=torch.device("cpu")))
model.eval()  # Switching to evaluation mode

# Generating test data
######################
n_test_images = 10

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

PSNR = 0
SSIM = 0
delta = 94
for i, data in enumerate(tqdm(test_loader)):
    # Get inputs and targets
    inputs, targets = data
    # Calculate outputs
    outputs = model(inputs)
    # Preprocess output tensors and targets for scikit-image
    outputs = outputs.detach().numpy()[:, 0, delta:-delta, delta:-delta]
    targets = targets.detach().numpy()[:, 0, delta:-delta, delta:-delta]
    # Calculate metrics
    for j in range(batch_size):
        output = outputs[j, ...]
        target = targets[j, ...]
        PSNR = PSNR + metrics.peak_signal_noise_ratio(target, output)/n_test_images
        SSIM = SSIM + metrics.structural_similarity(target, output)/n_test_images


print()
print("Testing results:")
print(f"PSNR: {PSNR}")
print(f"SSIM: {SSIM}")