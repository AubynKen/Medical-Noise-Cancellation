"""
File for evaluations functions used to evaluate the quality of denoising algorithms.
Author: Pinglei He
Credit: the PSNR functions is heavily based on the Python implementation of PSNR by Huang Liu
"""

import math
import numpy


def PSNR(img1, img2):
    """Calculate peak signal-to-noise ratio (PSNR) between two images."""
    arr_1 = numpy.array(img1, dtype=numpy.int64)
    arr_2 = numpy.array(img2, dtype=numpy.int64)
    diff = arr_1 - arr_2  # difference between the two images
    diff_squared = diff[:, :] ** 2  # square the difference
    rmse = diff_squared.sum() / img1.size  # calculate the root mean squared error
    psnr = 10 * math.log10(float(255. ** 2) / rmse)  # calculate the PSNR
    return psnr
