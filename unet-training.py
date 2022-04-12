from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm # progress bar
from torch.optim import Adam
from torchvision.utils import save_image
from torchvision import transforms
import argparse, yaml
from datetime import datetime
import warnings

# custom modules
from model import (StentDataset, # custom dataset
                   ImageAugmentation, # custom image generator
                   UNet) # our PyTorch U-Net model

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('-config', help="configuration file *.yml", type=str, required=False, default='unet-training.yml')
args = parser.parse_args()
args = yaml.load(open(args.config), Loader=yaml.FullLoader)

# flushing the GPU memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# creating checkpoint and results folders
exp_id = "unet_" + datetime.now().strftime("%Y%m%d_%H%M")
ckpt_path = os.path.join(args['ckpt_path'], exp_id)
result_path = os.path.join(args['results_path'], exp_id)
os.mkdir(ckpt_path)
os.mkdir(result_path)

# custom image generator
image_generator = ImageAugmentation(base_image_path=args['dataloader']['base_image_path'])

# generate train images
image_generator.generate_input_images(**args['dataloader']['train'])

# generate test images
image_generator.generate_input_images(**args['dataloader']['test'])

input_path = os.path.join(args['dataloader']['train']['save_path'], 'x')
target_path = os.path.join(args['dataloader']['train']['save_path'], 'y')
dataset = StentDataset(input_path, target_path)

model = UNet(in_channels=1, out_channels=1)
model.double()
training_loader = DataLoader(dataset, batch_size=args['dataloader']['batch_size'], shuffle=True)
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), **args['optimizer'])

model = model.cuda() if torch.cuda.is_available() else model

if torch.cuda.is_available() and not next(model.parameters()).is_cuda:
    warnings.warn("There is a CUDA device available, but the model was not successfuly sent to it.")

# parameters
epochs = args['epochs']

# training
losses = []
for epoch in range(epochs):
    for i, data in enumerate(tqdm(training_loader)):
        # get the inputs
        inputs, targets = data
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        # print statistics
        losses.append(loss.item())
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
              .format(epoch + 1, epochs, i + 1, len(training_loader), loss.item()))

    # save checkpoints
    if epoch % 50 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': LOSS,
            },
            os.path.join(ckpt_path, "ckpt_" + str(epoch))
        )

# saving the model
torch.save(model.state_dict(), os.path.join(result_path, "model.pt"))