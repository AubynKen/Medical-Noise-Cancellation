from unet import UNet
import torch
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

model = UNet(in_channels=1, out_channels=1)
model.load_state_dict(torch.load("../../weights/iter_09000.pt", map_location=torch.device("cpu")))


for f in os.listdir("../dataset/noisy_png"):
    im_arr = np.array(Image.open(os.path.join("../dataset/noisy_png", f)))
    im_arr /= (1 ** 16 - 1)
    model.double()
    im_tensor = torch.from_numpy(im_arr).unsqueeze(0).unsqueeze(0).double()
    output = model(im_tensor)
    output = output.detach().numpy()[0, 0, :, :]
    output[output < 0] = 0
    output[output > 1] = 1
    plt.imshow(output, cmap="gray")
    plt.show()