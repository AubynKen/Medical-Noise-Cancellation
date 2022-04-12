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

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('-config', help="configuration file *.yml", type=str, required=True, default='unet-training.yml')
args = parser.parse_args()
opt = yaml.load(open(args.config), Loader=yaml.FullLoader)
opt.update(vars(args))
args = opt


# custom modules
from model import (StentDataset, # custom dataset
                   ImageAugmentation, # custom image generator
                   UNet) # our PyTorch U-Net model

# custom image generator
image_generator = ImageAugmentation(base_image_path="data/dataset/base_png")

# generate train images
image_generator.generate_input_images(n_images=100,
                                      save_path="data/dataset/train",
                                      generate_input=True,
                                      print_progress=True)

# generate test images
image_generator.generate_input_images(n_images=20,
                                      save_path="data/dataset/test",
                                      generate_input=True,
                                      print_progress=False)

dataset = StentDataset(input_path="data/dataset/train/x",
                       target_path="data/dataset/train/y")

model = UNet(in_channels=1, out_channels=1)
model.double()
training_loader = DataLoader(dataset, batch_size=1, shuffle=True)
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay)

# initial prediction image
plt.imshow(dataset[26][0], cmap="gray")
prediction = model(dataset[26][0].reshape(1, 512, 512))
plt.imshow(prediction.detach().numpy()[0], cmap="gray")

# parameters
epochs = 1
batch_size = 1

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
torch.save(model.state_dict(), PATH)