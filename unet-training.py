from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm # progress bar
from torch.optim import Adam
import argparse, yaml
from datetime import datetime

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('-config', help="configuration file *.yml", type=str, required=True, default='unet-training.yml')
args = parser.parse_args()
args = yaml.load(open(args.config), Loader=yaml.FullLoader)

# custom modules
from model import (StentDataset, # custom dataset
                   ImageAugmentation, # custom image generator
                   UNet) # our PyTorch U-Net model

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

# initial prediction image
plt.imshow(dataset[26][0], cmap="gray")
prediction = model(dataset[26][0].reshape(1, 512, 512))
plt.imshow(prediction.detach().numpy()[0], cmap="gray")

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

# saving the model
model_save_path = os.path.join(args['model_save_path'], "unet_" + datetime.now().strftime("%Y%m%d_%H%M"))
torch.save(model.state_dict(), model_save_path)