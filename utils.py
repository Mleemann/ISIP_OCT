#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to crop image and retain only the retinal structures.
"""
import numpy as np
from skimage import filters


def create_mask(img, sigma=2):
    """
    creates a binary mask of the OCT image
    """
    blur = filters.gaussian(img, sigma)
    threshold = filters.threshold_otsu(blur)
    mask = blur > threshold

    return mask


def remove_border(img, mask):
    """
    removes the white boarder from the original image and the mask
    """
    border = np.where(mask == 0)
    top, down = (min(border[0]), max(border[0]))
    left, right = (min(border[1]), max(border[1]))

    img_no_border = img[top:down, left:right]
    mask_no_border = mask[top:down, left:right]

    return img_no_border, mask_no_border


def crop_image(img_no_border, mask_no_border):
    """
    crops image without border according to the mask
    to get only the retinal structure
    """
    box = np.where(mask_no_border[5:-5, 5:-5] == 1)
    top, down = (min(box[0]), max(box[0]))
    left, right = (min(box[1]), max(box[1]))

    cropped_img = img_no_border[top:down, left:right]

    return cropped_img
