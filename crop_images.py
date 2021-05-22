#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crop the images and retain only the retinal structures.
"""
import os
from skimage import color
import matplotlib.pyplot as plt
import utils as utls

path_train_srf = '/Users/Michele/PycharmProjects/ISIP_OCT/Train-Data/SRF/'
path_train_noSrf = '/Users/Michele/PycharmProjects/ISIP_OCT/Train-Data/NoSRF/'

images_srf = [os.path.basename(file) for file in os.listdir(path_train_srf)]
images_noSrf = [os.path.basename(file) for file in os.listdir(path_train_noSrf)]

cropped_srf = []
cropped_noSrf = []

for file in images_srf:
    os.chdir(path_train_srf)
    img = color.rgb2gray(plt.imread(file))

    mask = utls.create_mask(img)
    img_no_border, mask_no_border = utls.remove_border(img, mask)
    crop_img = utls.crop_image(img_no_border, mask_no_border)

    cropped_srf.append(crop_img)

for file in images_noSrf:
    os.chdir(path_train_noSrf)
    img = color.rgb2gray(plt.imread(file))

    mask = utls.create_mask(img)
    img_no_border, mask_no_border = utls.remove_border(img, mask)
    crop_img = utls.crop_image(img_no_border, mask_no_border)

    cropped_noSrf.append(crop_img)
