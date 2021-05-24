#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2 as cv
import os
from imageio import imread
import numpy as np
import preprocessing as pre
import utils as utls

######################
# Prepare input data #
######################
path_template_srf = '/Users/ibailertxundi/Desktop/patchSRF/'
path_template_no_srf = '/Users/ibailertxundi/Desktop/patchNoSRF/'
image_path = '/Users/ibailertxundi/Desktop/SRF/11.png'
#match_image = imread(image_path)

#list_templates_srf = [os.path.basename(file) for file in os.listdir(path_template_srf)]
#templates_srf = np.asarray(list_templates_srf)
#list_templates_no_srf = [os.path.basename(file) for file in os.listdir(path_template_no_srf)]
#templates_no_srf = np.asarray(list_templates_no_srf)

templates_no_srf = pre.preprocess_images(path_template_no_srf)
templates_srf = pre.preprocess_images(path_template_srf)
match_image = pre.preprocess(image_path)

# There are 6 matching options, we will most likely choose 'cv.TM_CCORR_NORMED' == 3
templeta_matching = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

m_method = eval(templeta_matching[3])


###################
# Build functions #
###################

# Cluster and choose best scores
def compare_scores(srf_scores,no_srf_scores,matching_img):
    # Assert array types and length
    assert 'list' in str(type(srf_scores))
    assert 'list' in str(type(no_srf_scores))
    assert len(no_srf_scores) == len(srf_scores)

    # Calculate the mean of each list type
    m_srf = sum(srf_scores) / len(srf_scores)
    m_no_srf = sum(no_srf_scores) / len(no_srf_scores)
    
    # Return list with image and its type
    if m_srf>m_no_srf:
        return(list(matching_img,'srf'))

    else:
        return(list(matching_img,'no_srf'))

# Matching algorithm over the image with all templates
def matching(templates, img_to_match, matching_method):
    # Create empty list to output
    best_scores = list()
    for patch in templates:
        scores = cv.matchTemplate(img_to_match,patch, matching_method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(scores)
        best_scores.append(max_val)
    
    return(best_scores)

################
# Run matching #
################

# Get best scores
best_sfr = matching(templates_srf,match_image,m_method)

best_no_sfr = matching(templates_no_srf,match_image,m_method)

# Get list with image and assigned type
Result = compare_scores(best_sfr,best_no_sfr,match_image)

print(Result[2])
