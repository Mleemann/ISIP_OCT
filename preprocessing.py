#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessing images for training or testing. Program includes cropping of
the images, histogram equalization and ?non-local mean filtering?
"""
import utils as utls


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
