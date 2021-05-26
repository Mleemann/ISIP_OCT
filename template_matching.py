#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2 as cv
import os
from imageio import imread
import numpy as np
import preprocessing as pre
import utils as utls
import matplotlib.pyplot as plt
import skimage

###################
# Build functions #
###################

# Cluster and choose best scores
def compare_scores(srf_scores,no_srf_scores,matching_img):
    """
    Cluster and choose best scores

    input: matching image, srf and no srf scores to compare
    output: list with image and result
    """
    # Assert array types and length
    assert 'list' in str(type(srf_scores))
    assert 'list' in str(type(no_srf_scores))
    assert len(no_srf_scores) == len(srf_scores)

    # Calculate the mean of each list type
    m_srf = sum(srf_scores) / len(srf_scores)
    m_no_srf = sum(no_srf_scores) / len(no_srf_scores)
    
    # Return list with image and its type
    if m_srf>m_no_srf:
        return([matching_img,1])

    else:
        return([matching_img,0])


def matching(templates, img_to_match, matching_method):
    """
    Implement matching algorithm over the image with all templates

    input: template images, images to match and method
    output: list with best scores
    """
    # Create empty list to output
    best_scores = list()
    for patch in templates:
        scores = cv.matchTemplate(img_to_match,patch, matching_method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(scores)
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if matching_method in ['cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']:
            best_scores.append(min_val)
        else:
            best_scores.append(max_val)
    
    return(best_scores)


def convert(images):
    """
    Transform images to 8-bit

    input: images
    output: transformed images
    """
    converted_images = []
    for image in images:
        converted = skimage.img_as_ubyte(image)
        converted_images.append(converted)
    
    return converted_images


