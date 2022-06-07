
For details description of code, please see the description of the program in the corresponding files.

## `code` contains programs used for processing the image.
- `bm3d_result.py`: generation of de-noised images using bm3d
- `contrast_adjustment.py`: generation of de-noised images, adjusting the contrast to be the same as noisy images
- `image_resolution.py`: enhancement of the quality of base 1 to base 6 images
- `segmentation.py`: creation of masks for segmentation purposes

## `dataset` contains data used for training and evaluation
- `base_png_8bit`: base images initially encoded in 8 bits converted into 16bit png.
- `base_png`: base images after quality enhancement.
- `noisy_png`: noisy images with gaussian noise with standard deviation of 4 times the std of the base images.
- `base_mask`: serialized numpy arrays of masks for base images.
- `base_mask_png`: base masks converted to images for visualization.

## `result` contains results of the experiments
- `bm3d`: results denoising using bm3d.