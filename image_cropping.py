from PIL import Image
from image_paths import path_orig_tif

# Opens a image
im = Image.open(path_orig_tif)

# Size of the image in pixels
width, height = im.size  # 1024 * 1024

# Setting the points for cropped image
left = 0
top = int(2 * height / 3)
right = int(width / 3)
bottom = height

# Cropped image of above dimension
im1 = im.crop((left, top, right, bottom))

# Shows the image in image viewer
im1.show()
