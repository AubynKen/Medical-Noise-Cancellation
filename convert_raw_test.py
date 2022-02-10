from matplotlib import pyplot as plt
import numpy as np
image = np.empty((1024, 1024), np.uint16)
image.data[:] = open('./data/raw/CDStent_noise4.raw').read()
plt.imshow(image)
