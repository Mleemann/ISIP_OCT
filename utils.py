#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to crop image and retain only the retinal structures.
"""
import numpy as np
from skimage import filters, color, exposure
import os
import matplotlib.pyplot as plt
from skimage.restoration import denoise_bilateral, denoise_nl_means


def create_mask(img, sigma=2):
    """
    creates a binary mask of the OCT image

    input: image
    output: mask of the image
    """
    blur = filters.gaussian(img, sigma)
    threshold = filters.threshold_otsu(blur)
    mask = blur > threshold

    return mask


def remove_border(img, mask):
    """
    removes the white border from the original image and the mask

    input: original image and its mask
    output: image and mask without the white border
    """
    border = np.where(mask == 0)
    top, down = (min(border[0]), max(border[0]))
    left, right = (min(border[1]), max(border[1]))

    img_no_border = img[top:down, left:right]
    mask_no_border = mask[top:down, left:right]

    return img_no_border, mask_no_border


def crop(img_no_border, mask_no_border):
    """
    crops image without border according to the mask
    to get only the retinal structure

    input: image without any white border and its mask
    output: cropped image with only retinal layer
    """
    box = np.where(mask_no_border[5:-5, 5:-5] == 1)
    top, down = (min(box[0]), max(box[0]))
    left, right = (min(box[1]), max(box[1]))

    cropped_img = img_no_border[top:down, left:right]

    return cropped_img


def crop_images(path_to_images):
    """
    crops all the images within a folder

    input: path to the folder with images to crop
    output: list of cropped images
    """
    images = [os.path.basename(file) for file in os.listdir(path_to_images)]
    cropped_images = []

    for file in images:
        os.chdir(path_to_images)
        img = color.rgb2gray(plt.imread(file))

        mask = create_mask(img)
        img_no_border, mask_no_border = remove_border(img, mask)
        crop_img = crop(img_no_border, mask_no_border)

        cropped_images.append(crop_img)

    return cropped_images



def histogram_equalization(image):
    """
    Adjusts contrasts of an image using it's histogram

    input: image
    output: image after histogram equalization
    """
    img_eq = exposure.equalize_hist(image)

    return img_eq


def bilateral_denoising(image):
    """
    Removes noise from an image by linear filtering

    input: image
    output: image after denoising
    """
    img_denoised = denoise_bilateral(image)

    return img_denoised


def nlm_denoising(image):
    """
    Removes noise from an image by non-local mean filtering

    input: image
    output: image after denoising
    """
    img_denoised = denoise_nl_means(image)

    return img_denoised
