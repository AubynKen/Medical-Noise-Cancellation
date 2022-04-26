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

    def __getitem__(self, idx):
        return self._getitem_offline(idx)


class StentOnlineDataset(Dataset):
    """
    Dataset for online training.
    Generates input and output images for the unet during training without taking extra space in disk.
    """

    # parameters of the noise, hardcoded for now
    mu = 0  # mean of noise, gaussian noise centered at zero
    base_std = 1398.96558145  # standard deviation of un-cropped stent image i.e. CDStent.png in 16 bits
    sigma = 4 * base_std  # std of noise, arbitrary choice

    def __init__(self, n_images: int, base_image_path: str):
        self.n_images = n_images
        self.select_base_img = lambda: select(base_image_path)

    def __len__(self):
        return self.n_images

    def _get_numpy_arr_after_pipeline(self, idx):
        # check if the index is valid
        if idx >= self.n_images:
            raise IndexError("Index out of range")
        # get the base image
        base_img = self.select_base_img()
        # transformation
        y_img = pipeline(base_img)  # apply the pipeline to the base image
        y_arr = np.array(y_img, dtype="float")  # convert to numpy array
        return y_arr

    def _generate_training_data(self, y_arr, std_factor=4):
        # add noise to create the input image
        x_arr = y_arr / 3 + np.random.normal(loc=self.mu, scale=std_factor * self.base_std, size=y_arr.shape)
        # normalization
        y_arr, x_arr = y_arr / 2 ** 16, x_arr / 2 ** 16
        # crop the target image to fit the correct output size
        delta = 94  # we loose 94 pixels on each side during convolution
        y_arr = y_arr[delta:-delta, delta:-delta]
        # convert to pytorch tensors
        target_tensor = torch.tensor(y_arr, dtype=torch.double)
        input_tensor = torch.tensor(x_arr, dtype=torch.double)
        # unsqueeze the tensor
        target_tensor = target_tensor.unsqueeze(0)
        input_tensor = input_tensor.unsqueeze(0)
        # send to cuda if cuda available
        if torch.cuda.is_available():
            target_tensor = target_tensor.cuda()
            input_tensor = input_tensor.cuda()
        return input_tensor, target_tensor

    def __getitem__(self, idx):
        y_arr = self._get_numpy_arr_after_pipeline(idx)
        input_tensor, target_tensor = self._generate_training_data(y_arr)
        return input_tensor, target_tensor


class StentOnlineDatasetRandomSTD(StentOnlineDataset):
    """
    Dataset for online training with random noise standard deviation.
    """

    def __init__(self, n_images: int, base_image_path: str, std_range: [float, float]):
        """
        :param n_images: number of images to generate
        :param base_image_path: path to the base image
        :param std_range: range of standard deviations factor. A factor of one means that the noise has the same std as
        the base image.
        """
        super().__init__(n_images, base_image_path)
        min_std_factor, max_std_factor = std_range
        if min(min_std_factor, max_std_factor) < 0:
            raise ValueError("std_range must be positive")
        self.min_std = min_std_factor
        self.max_std = max_std_factor

    def __getitem__(self, idx):
        y_arr = self._get_numpy_arr_after_pipeline(idx)
        std_factor = np.random.uniform(self.min_std, self.max_std)
        input_tensor, target_tensor = self._generate_training_data(y_arr, std_factor)
        return input_tensor, target_tensor
