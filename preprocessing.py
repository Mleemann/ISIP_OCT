#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessing images for training or testing. Program includes cropping of
the images, histogram equalization and ?non-local mean filtering?
"""
import utils as utls
import os
from skimage import filters, color, exposure
import matplotlib.pyplot as plt


def preprocess_images(path_to_images):
    """
    input: path to the folder with the images to be processed
    output: list of cropped, denoised images
    """
    cropped_images = utls.crop_images(path_to_images)
    denoised_images = []

    for image in cropped_images:
        eq_img = utls.histogram_equalization(image)
        denoised_img = utls.nlm_denoising(eq_img)
        denoised_images.append(denoised_img)

    return denoised_images


def preprocess(path_to_images):
    """
    input: path to the folder with the images to be processed
    output: list of preprocessed images
    """
    images_original = [os.path.basename(file) for file in os.listdir(path_to_images)] 
    images =[]
    for file in images_original:
        os.chdir(path_to_images)
        img = color.rgb2gray(plt.imread(file))
        images.append(img)
     
    denoised_images = []

    for image in images:
        eq_img = utls.histogram_equalization(image)
        denoised_img = utls.nlm_denoising(eq_img)
        denoised_images.append(denoised_img)

    return denoised_images