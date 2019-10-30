import os
import math
import numpy as np
import mahotas as mh
import matplotlib.pylab as plt
import cv2

L = 0.5
S = 0.48
W = 12.14
LPF = 0.001
Threshold_min = -1
Threshold_max = 0.0019
FLAG = 1


def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho


def phase_stretch_transform(img, LPF, S, W, threshold_min, threshold_max, flag):
    L = 0.5
    x = np.linspace(-L, L, img.shape[0])
    y = np.linspace(-L, L, img.shape[1])
    [X1, Y1] = (np.meshgrid(x, y))
    X = X1.T
    Y = Y1.T
    theta, rho = cart2pol(X, Y)
    orig = ((np.fft.fft2(img)))
    expo = np.fft.fftshift(np.exp(-np.power((np.divide(rho, math.sqrt((LPF ** 2) / np.log(2)))), 2)))
    orig_filtered = np.real(np.fft.ifft2((np.multiply(orig, expo))))
    PST_Kernel_1 = np.multiply(np.dot(rho, W), np.arctan(np.dot(rho, W))) - 0.5 * np.log(
        1 + np.power(np.dot(rho, W), 2))
    PST_Kernel = PST_Kernel_1 / np.max(PST_Kernel_1) * S
    temp = np.multiply(np.fft.fftshift(np.exp(-1j * PST_Kernel)), np.fft.fft2(orig_filtered))
    orig_filtered_PST = np.fft.ifft2(temp)
    PHI_features = np.angle(orig_filtered_PST)
    if flag == 0:
        out = PHI_features
    else:
        features = np.zeros((PHI_features.shape[0], PHI_features.shape[1]))
        features[PHI_features > threshold_max] = 1
        features[PHI_features < threshold_min] = 1
        features[img < (np.amax(img) / 20)] = 0

        out = features
        out = mh.thin(out, 1)
        out = mh.bwperim(out, 4)
        out = mh.thin(out, 1)
        out = mh.erode(out, np.ones((1, 1)))
    return out, PST_Kernel


def main():
    Image_orig = mh.imread("./cameraman.tif")
    if Image_orig.ndim == 3:
        Image_orig_grey = mh.colors.rgb2grey(Image_orig)
    else:
        Image_orig_grey = Image_orig
    edge, kernel = phase_stretch_transform(Image_orig_grey, LPF, S, W, Threshold_min,
                                           Threshold_max, FLAG)
    Overlay = mh.overlay(Image_orig_grey, edge)
    Edge = edge.astype(np.uint8) * 255
    plt.imshow(Edge)
    plt.show()
