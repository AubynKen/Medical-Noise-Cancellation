"""
Segmentation of the base images into binary pixels, where a white pixel represents the background and a black pixel
represents the stents.
"""

from PIL import Image
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

def segment_images(image_path, masked_image_path, mask_path):
    """
    Segment the images into binary pixels, where a white pixel represents the background and a black pixel
    represents the stents.
    :param image_path: Path to the image to be segmented.
    :param masked_image_path: Path to segmented image to be saved
    :param mask_path: Path to the mask to be saved
    """
    # Load the image
    image = Image.open(image_path)
    # Convert the image to a numpy array
    image_array = np.array(image)
    # maximum intensity value
    max_pixel = np.max(image_array)
    # mask the image
    mask = image_array < max_pixel
    # convert mask to image
    mask_image = Image.fromarray(mask.astype(np.uint16) * 65535)
    # Save the segmented image
    mask_image.save(masked_image_path)
    # save the mask
    np.save(mask_path, mask)

for f in os.listdir("../dataset/base_png"):
    segment_images(os.path.join("../dataset/base_png", f),
                   os.path.join("../dataset/base_mask_png", f),
                   os.path.join("../dataset/base_mask", f))



