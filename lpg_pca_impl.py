import numpy as np
from multiprocessing import Pool
import os
from sklearn.feature_extraction import image


def _denoise_pixel(img, x, y, K, L, sig):
    def getBlock(x, y):
        return img[x - halfK: x + halfK + 1, y - halfK: y + halfK + 1]

    # def mse(block):
    #     return np.mean((block - target)**2)
    halfK = K//2
    halfL = L//2
    # Dimension of each block vector (= number of rows in the training matrix)
    m = K**2

    # Number of columns in the training matrix
    n = m * 8 + 1

    # Block centered around x,y
    target = getBlock(x, y)

    # Assemble a pool of blocks.
    dim1, dim2 = img.shape
    rng = halfL - halfK
    blocks = image.extract_patches(
        img[max(K, x - rng) - halfK: min(x + rng + 1, dim2 - K) + halfK,
            max(K, y - rng) - halfK: min(y + rng + 1, dim1 - K) + halfK], (K, K)
    ).reshape(-1, K, K)

    # Sort by MSE
    sortIndexes = ((blocks - target) **
                   2).reshape(blocks.shape[0], m, order='F').mean(axis=1).argsort()

    # Construct the training matrix with the target and the best blocks reshaped into columns.
    trainingMatrix = blocks[sortIndexes].reshape(
        blocks.shape[0], m, order='F').swapaxes(1, 0)[:, :n+1]

    mean = trainingMatrix.mean(axis=1)
    trainingMatrix = trainingMatrix - mean.reshape(m, 1)
    noiseCov = sig**2 * np.eye(m, m)
    inputCov = (trainingMatrix @ trainingMatrix.T)/n
    eigvectors = np.linalg.eig(inputCov)[1]
    PX = eigvectors.T

    transInput = PX @ trainingMatrix

    transNoiseCov = PX @ noiseCov @ PX.T
    transInputCov = (transInput @ transInput.T)/n
    transDenoisedOutCov = np.maximum(
        np.zeros(transInputCov.shape), transInputCov - transNoiseCov)

    shrinkCoef = np.diag(transDenoisedOutCov) / \
        (np.diag(transDenoisedOutCov) + np.diag(transInputCov))
    Y1 = transInput[:, 0] * shrinkCoef
    X1 = PX.T @ Y1 + mean
    return X1[m//2]


def _denoise_row(img, x, left_y, right_y, K, L, sig, log):
    if log:
        print(x)
    return (x, left_y, right_y,
            [_denoise_pixel(img, x, y, K, L, sig) for y in range(left_y, right_y)])


def _denoise_image(img, K, L, sig, log):
    global outImg

    outImg = np.copy(img)
    width, height = img.shape
    halfL = L // 2
    halfK = K // 2

    def denoiseRowCallback(result):
        global outImg

        x, y_left, y_right, data = result
        outImg[x, y_left:y_right] = data

    global pool

    # parallel
    progress = [pool.apply_async(_denoise_row, (img, x, halfK, height - halfK, K, L,
                                                sig, log,), callback=denoiseRowCallback) for x in range(halfK, width - halfK)]
    for each in progress:
        each.wait()

    # non-parallel:
    # for x in range(halfK, width - halfK):
    #     if log:
    #         print(x)
    #     for y in range(halfK, height - halfK):
    #         outImg[x, y] = _denoise_pixel(img, x, y, K, L, sig)

    return outImg


def denoise(noised_img, sig1, K=3, L=21, log=False):
    global pool

    try:
        pool  # pool already exists
    except NameError:
        # creating new pool
        # don't use all cores, your UI may start to lag
        pool = Pool(os.cpu_count() - 1)

    stage1 = _denoise_image(noised_img, K, L, sig1, log)

    sig2 = 0.35 * np.sqrt(sig1 - np.mean((stage1 - noised_img)**2))
    if log:
        print('sig2 = ', sig2)

    stage2 = _denoise_image(stage1, K, L, sig2, log)

    # pool.terminate()

    return stage2
