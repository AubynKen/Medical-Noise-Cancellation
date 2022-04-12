from PIL import Image
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


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

    def _getitem_offline(self, idx):
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
        target_image = target_img[delta:-delta, delta:-delta]
        # normalize the image
        input_img = input_img / 2 ** 16
        target_image = target_image / 2 ** 16
        # return the image
        return input_img, target_image

    def __getitem__(self, idx):
        return self._getitem_offline(idx)
