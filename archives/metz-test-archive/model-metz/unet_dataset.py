from PIL import Image
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .input_data_augmentation import pipeline, select


class StentDataset(Dataset):
    def __init__(self, input_path, target_path):
        self.input_path = input_path
        self.target_path = target_path
        self.input_files = os.listdir(self.input_path)
        self.target_files = os.listdir(self.target_path)
        self.input_files.sort()
        self.target_files.sort()

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        # files corresponding to the index
        input_file = self.input_files[idx]
        target_file = self.target_files[idx]
        # load the image
        input_img = Image.open(os.path.join(self.input_path, input_file))
        target_img = Image.open(os.path.join(self.target_path, target_file))
        # convert to numpy array
        input_img = torch.tensor(np.array(input_img), dtype=torch.double)
        target_img = torch.tensor(np.array(target_img), dtype=torch.double)
        # crop the image
        delta = 94  # we loose 94 pixels on each side during convolution
        target_img = target_img[delta:-delta, delta:-delta]
        # normalize the image
        input_img = input_img / 2 ** 16
        target_img = target_img / 2 ** 16
        # unsqueeze the images
        input_img = input_img.unsqueeze(0)
        target_img = target_img.unsqueeze(0)
        # send to cuda device if available
        input_img = input_img.cuda() if torch.cuda.is_available() else input_img
        target_img = target_img.cuda() if torch.cuda.is_available() else target_img
        return input_img, target_img


class StentOnlineDataset(Dataset):
    """
    Dataset for online training.
    Generates input and output images for the unet during training without taking extra space in disk.
    """

    # parameters of the noise, hardcoded for now
    mu = 15802.056997617085  # mean
    sigma = 5595.862325808515  # standard deviation

    def __init__(self, n_images: int, base_image_path: str):
        self.n_images = n_images
        self.select_base_img = lambda: select(base_image_path)

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        # check if the index is valid
        if idx >= self.n_images:
            raise IndexError("Index out of range")
        # get the base image
        base_img = self.select_base_img()
        # transformation
        y_img = pipeline(base_img)  # apply the pipeline to the base image
        y_arr = np.array(y_img, dtype="float")  # convert to numpy array
        # add noise to create the input image
        x_arr = y_arr / 3 + np.random.normal(loc=self.mu, scale=self.sigma, size=y_arr.shape)
        # normalization
        y_arr, x_arr = y_arr / 2 ** 16, x_arr / 2 ** 16
        # crop the target image to fit the correct output size
        delta = 94  # we loose 94 pixels on each side during convolution
        y_arr = y_arr[delta:-delta, delta:-delta]
        # convert to pytorch tensors
        target_tensor = torch.tensor(y_arr, dtype=torch.double)
        input_tensor = torch.tensor(x_arr, dtype=torch.double)
        return input_tensor, target_tensor
