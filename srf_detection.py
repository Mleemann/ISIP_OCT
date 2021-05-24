import glob
import csv
import cv2 as cv
import os
from imageio import imread
import template_matching as tmpmatch


def main():
    ######################
    # Prepare input data #
    ######################
    path_template_srf = '/Users/tugbaucar/Desktop/ISIP_OCT/Train-Data/SRF'
    path_template_no_srf = '/Users/tugbaucar/Desktop/ISIP_OCT/Train-Data/NoSRF'
    image_path = '/Users/tugbaucar/Desktop/ISIP_OCT/Train-Data/SRF'   #will become : '/Users/tugbaucar/Desktop/ISIP_OCT/Test-Data/handout'
    
    templates_no_srf = pre.preprocess_images(path_template_no_srf)
    
    templates_srf = pre.preprocess_images(path_template_srf)
    match_image = pre.preprocess(image_path)
    
    # There are 6 matching options, we will most likely choose 'cv.TM_CCORR_NORMED' == 3
    templeta_matching = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
    
    result_filename = 'project_Leemann_Lertxundi_Agaoglu.csv'    
    
    ################
    # Run matching #
    ################
    
    # Get best scores
    best_sfr = tmpmatch.matching(templates_srf,match_image,m_method)
    best_no_sfr = tmpmatch.matching(templates_no_srf,match_image,m_method)
    
    # Get list with image and assigned type
    image_names, img_classes = tmpmatch.compare_scores(best_sfr,best_no_sfr,match_image)

    # create csv output file
    write_csv(image_names, img_classes, result_filename)
    
def write_csv(image_names, img_classes, filename):
    """Produce csv output file from a list of image names and classification."""
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filename', 'label'])
        writer.writerows(zip(image_names, img_classes))

if __name__ == '__main__':
    main()
