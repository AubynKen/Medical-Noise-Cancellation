"""
Denoise noisy images from validation set for comparison.
Author: Pinglei He
"""

from evaluation import denoise_BM3D
import os

path_noisy_images = "../dataset/noisy_png"
for filename in os.listdir(path_noisy_images):
    denoise_BM3D(noisy_img_path=os.path.join(path_noisy_images, filename),
                 save_path=os.path.join("../dataset/results/bm3d", filename))
