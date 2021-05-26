"""
OCT image SRF detection

Classification of OCT images for the retinal disease bio-marker SRF (sub-retinal fluid).
This is the file to run for completing the task and output the classification of the
given images.

"""

import glob
import csv
import cv2 as cv
import os
import numpy as np
from imageio import imread
import utils as utls
import matplotlib.pyplot as plt
import skimage
import template_matching as tmpmatch
import preprocessing as pre


def main():
    ######################
    # Prepare input data #
    ######################
    path_template_srf = '/Users/tugbaucar/Desktop/ISIP_OCT/Train-Data/SRF/'
    path_template_no_srf = '/Users/tugbaucar/Desktop/ISIP_OCT/Train-Data/NoSRF/'
    image_path = '/Users/tugbaucar/Desktop/ISIP_OCT/Test-Data/handout/'   #will become : '/Users/tugbaucar/Desktop/ISIP_OCT/Test-Data/handout'
    
    image_paths = glob.glob(image_path + '*')
    image_names = [os.path.basename(image_path) for image_path in image_paths]
    
    result_filename = '/Users/tugbaucar/Desktop/ISIP_OCT/project_Leemann_Lertxundi_Agaoglu.csv' 
    
    # There are 6 matching options, we will most likely choose 'cv.TM_CCORR_NORMED' == 3
    templeta_matching = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
    
    m_method = eval(templeta_matching[3]) 
    
    
    templates_no_srf = pre.preprocess_images(path_template_no_srf)    
    templates_srf = pre.preprocess_images(path_template_srf)
    match_images = pre.preprocess(image_path)

    
    ################
    # Run matching #
    ################
    
    image_classes = []
    
    for match_image in match_images:
    
        # Convert images to 8-bit and numpy array
        templates_no_srf = np.asarray(tmpmatch.convert(templates_no_srf))
        templates_srf = np.asarray(tmpmatch.convert(templates_srf))
    
        match_image = np.asarray(tmpmatch.convert(match_image))
    
        # Get best scores
        best_sfr = tmpmatch.matching(templates_srf,match_image,m_method)
        best_no_sfr = tmpmatch.matching(templates_no_srf,match_image,m_method)
    
        # Get list with image and assigned type
        Result = tmpmatch.compare_scores(best_sfr,best_no_sfr,match_image)
    
        image_classes.append(Result[1])
        

    # create csv output file
    write_csv(image_names, image_classes, result_filename)
    

def write_csv(image_names, image_classes, filename):
    """Produce csv output file from a list of image names and classification."""
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filename', 'label'])
        writer.writerows(zip(image_names, image_classes))


if __name__ == '__main__':
    main()
