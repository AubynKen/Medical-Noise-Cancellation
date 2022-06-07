# code for our data augmentation pipeline
# author: Pinglei He

from PIL import Image, ImageChops, ImageOps
import numpy as np
import random as rd
import os
from math import ceil
import uuid  # for random image id generation

# parameters
pixel_max = 2 ** 16 - 1  # 16 bit unsigned integers
im_size = 512  # 512 * 512 pixels per base


# pipeline creation
def create_pipeline(*operations):
    """Select base image using selector function, and return the image after all transformations functions"""

    def pipeline(img: Image.Image):
        for op in operations:
            img = op(img)
        return img

    return pipeline


# function for selecting base image
def select(base_path) -> Image.Image:
    """Selects a random base image."""
    base_files = os.listdir(base_path)
    selected = rd.choice(base_files)
    return Image.open(os.path.join(base_path, selected))


# functions for applying transformations
def rotate(img: Image.Image, angle=None) -> Image.Image:
    if angle is None:
        angle = rd.uniform(0, 360)
    return img.rotate(angle, fillcolor=pixel_max)


def rescale(img: Image.Image, scale=None) -> Image.Image:
    if not scale:
        scale = rd.uniform(0.5, 1.8)
    # shrink the image
    thumbnail_size = ceil(im_size * scale / 2) * 2

    # add border to have the original image dimensions if image is shrunken
    if thumbnail_size <= im_size:
        img = img.resize((thumbnail_size, thumbnail_size))
        border_size = (im_size - thumbnail_size) // 2
        return ImageOps.expand(img, border=border_size, fill=pixel_max)

    # crop image if image is enlarged
    border_size = (thumbnail_size - im_size) // 2
    img = img.resize((thumbnail_size, thumbnail_size))
    return img.crop((border_size, border_size, border_size + im_size, border_size + im_size))


def translate(img: Image.Image, tx=None, ty=None) -> Image.Image:
    """
    :param img: image file
    :param tx: the number of pixels the image is translated to the left
    :param ty: the number of pixels the image is translated to the right
    :return: the translated image
    """
    # if tx or ty set to none, randomly shift by -200 to 200 pixels in both directions
    if not tx:
        tx = rd.randint(-100, 100)
    if not ty:
        ty = rd.randint(-100, 100)
    return ImageChops.offset(img, tx, ty, )


def flip_and_mirror(img: Image.Image) -> Image.Image:
    if np.random.uniform() >= 0.5:  # flip with a probability of 0.5
        img = ImageOps.flip(img)
    if np.random.uniform() >= 0.5:  # mirror with a probability of 0.5
        img = ImageOps.mirror(img)
    return img


# pipeline creation
pipeline = create_pipeline(flip_and_mirror, rotate, rescale, translate)


class ImageAugmentation:
    def __init__(self, base_image_path: str):
        """
        :param base_image_path: path to the folder containing the base images
        """
        self.base_path = base_image_path

    def generate_input_images(self, n_images: int, save_path: str, generate_input: bool = False,
                              print_progress=True, 
                              input_dir='x', gt_dir='y') -> None:
        """
        :param n_images: number of images to create
        :param save_path: path to the folder where the images will be saved
        :param generate_input: if True, the input images with noise will be created
        :param print_progress: if True, print the progress of the image generation
        :param input_dir: name of the input directory
        :param gt_dir: name of the ground truth directory
        """

        save_path_y = os.path.join(save_path, 'y') if generate_input else save_path
        # create directory if it does not exist
        if not os.path.exists(save_path_y):
            os.makedirs(save_path_y)
        else:
            print("The data path already has a 'y' directory! Please select a path that doesn't contain an output nor "
                  "input directory yet.")
            return

        # generate input images
        progress_percentage = 0
        print(f"Generating {n_images} output images...")

        # disable print progress if less than 100 images
        if n_images < 100:
            print_progress = False
            print("Print progress disabled since less than 100 images are generated.")

        for i in range(n_images):
            if print_progress and i % (n_images // 100) == 0:
                progress_percentage += 1
                print(f"{i}/{n_images} output images generated.  ", end="")
                print(f"Progress: {progress_percentage}%",  end="\r")
            # create random image id
            image_id = str(uuid.uuid4())
            # select base image
            base_img = select(self.base_path)
            # apply transformations
            img = pipeline(base_img)
            # save image
            img.save(os.path.join(save_path_y, image_id + ".png"))
        print(f"{n_images} output images generated at {save_path_y}")

        if not generate_input:
            return

        # parameters of the noise
        mu = 15802.056997617085  # mean
        sigma = 5595.862325808515  # standard deviation

        # generate output images
        print(f"Generating {n_images} input images with noise...")
        progress_percentage = 0
        original_images = os.listdir(save_path_y)
        save_path_x = os.path.join(save_path, 'x')

        # create directory for output images if it does not exist
        if not os.path.exists(save_path_x):
            os.makedirs(save_path_x)

        # noised image generation
        for i, image in enumerate(original_images):
            if print_progress and i % (n_images // 100) == 0:
                progress_percentage += 1
                print(f"{i}/{n_images} input images generated.  ", end="")
                print(f"Progress: {progress_percentage}%",  end="\r")
            im_orig = Image.open(os.path.join(save_path_y, image))
            im_arr = np.array(im_orig, dtype="float")
            im_arr_noised = im_arr / 3 + np.random.normal(loc=mu, scale=sigma, size=im_arr.shape)
            im_noised = Image.fromarray(im_arr_noised.astype(np.uint16))
            im_noised.save(os.path.join(save_path_x, image))
        print(f"{n_images} input images generated at {save_path_x}")
