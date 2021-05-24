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

######################
# Prepare input data #
######################
path_template_srf = '/Users/ibailertxundi/Desktop/SRF/'
path_template_no_srf = '/Users/ibailertxundi/Desktop/NoSRF/'
image_path = '/Users/ibailertxundi/Desktop/rest_of_images/nosrf/'

templates_no_srf = pre.preprocess_images(path_template_no_srf)
templates_srf = pre.preprocess_images(path_template_srf)
match_images = pre.preprocess(image_path)
match_image = match_images[1]

# There are 6 matching options, we will most likely choose 'cv.TM_CCORR_NORMED' == 3
templeta_matching = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

m_method = eval(templeta_matching[3])


###################
# Build functions #
###################

# Cluster and choose best scores
def compare_scores(srf_scores,no_srf_scores,matching_img):
    """
    # Cluster and choose best scores

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
        return([matching_img,'srf'])

    else:
        return([matching_img,'no_srf'])


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

################
# Run matching #
################

# Convert images to 8-bit and numpy array
templates_no_srf = np.asarray(convert(templates_srf))
templates_srf = np.asarray(convert(templates_srf))
match_image = np.asarray(convert(match_image))

# Get best scores
best_sfr = matching(templates_srf,match_image,m_method)
best_no_sfr = matching(templates_no_srf,match_image,m_method)

# Get list with image and assigned type
Result = compare_scores(best_sfr,best_no_sfr,match_image)

print(Result[1])
