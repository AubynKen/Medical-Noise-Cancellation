# Purposes of different folders
- **base_mask_png**: binary masks of base images for segmentation purposes. Segmentation was not used in our project but
the images were kept for archiving purposes.
- **base_png**: base images without noise provided by General Electric Healthcare
- **base_png_8bit**: we've used 16 bit images during our training. This folder contains images in 8 bit provided by 
Electric Healthcare before we converted them into 16 bits for compatibility issues.
- **noisy_png**: base images with gaussian noise.
- **more_base_png**: stent images that we've found on the Internet