"""
File for evaluations functions used to evaluate the quality of denoising algorithms.
Author: Pinglei He, Kaiao Wen, Yassine
Credit: the PSNR functions is heavily based on the Python implementation of PSNR by Huang Liu
"""

import math
import numpy as np
import cv2


def PSNR(img1, img2):
    """Calculate peak signal-to-noise ratio (PSNR) between two images."""
    arr_1 = np.array(img1, dtype=np.int64)
    arr_2 = np.array(img2, dtype=np.int64)
    diff = arr_1 - arr_2  # difference between the two images
    diff_squared = diff[:, :] ** 2  # square the difference
    rmse = diff_squared.sum() / img1.size  # calculate the root mean squared error
    psnr = 10 * math.log10(float(255. ** 2) / rmse)  # calculate the PSNR
    return psnr

def ssim(img1, img2, L):
    """Calculate SSIM (structural similarity) for one channel images.
    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
    Returns:
        float: ssim result.
    """
    K1 = 0.01
    K2 = 0.03
    C1 = (K1 * L)**2
    C2 = (K2 * L)**2
    C3 = C2/2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    # ux
    ux = img1.mean()
    # uy
    uy = img2.mean()
    # ux^2
    ux_sq = ux**2
    # uy^2
    uy_sq = uy**2
    # ux*uy
    uxuy = ux * uy
    # ox、oy方差计算
    ox_sq = img1.var()
    oy_sq = img2.var()
    ox = np.sqrt(ox_sq)
    oy = np.sqrt(oy_sq)
    oxoy = ox * oy
    oxy = np.mean((img1 - ux) * (img2 - uy))
    # 公式一计算
    L = (2 * uxuy + C1) / (ux_sq + uy_sq + C1)
    C = (2 * ox * oy + C2) / (ox_sq + oy_sq + C2)
    S = (oxy + C3) / (oxoy + C3)
    ssim = L * C * S
    # 验证结果输出
    # print('ssim:', ssim, ",L:", L, ",C:", C, ",S:", S)
    return ssim

def _ssim(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images.
    It is called by func:`calculate_ssim`.
    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.
    Returns:
        float: ssim result.
    """

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
	# 公式二计算
    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()