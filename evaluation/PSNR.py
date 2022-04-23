import math
import numpy


def PSNR(img1, img2):
    D = numpy.array(img1 - img2, dtype=numpy.int64)
    D[:, :] = D[:, :] ** 2
    RMSE = D.sum() / img1.size
    psnr = 10 * math.log10(float(255. ** 2) / RMSE)
    return psnr