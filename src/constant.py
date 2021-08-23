#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 01:19:41 2021

@author: recepaydogdu
"""
import os

DATA_DIR = "../data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
MASK_POLY_DIR = os.path.join(DATA_DIR, "masks_poly") #Mask'ların kaydedileceği dosya yolu.
MASK_LINE_DIR = os.path.join(DATA_DIR, "masks_line")
JSON_DIR = os.path.join(DATA_DIR, "jsons") #Annotation dosyalarının dosya yolu.
MASKED_IMAGES_DIR = os.path.join(DATA_DIR, "masked_images")
MODELS_DIR = "../models"
PREDICT_DIR = os.path.join(DATA_DIR, "predicts")
AUG_IMAGE_DIR = os.path.join(DATA_DIR, "aug_image")
AUG_MASK_DIR = os.path.join(DATA_DIR, "aug_mask")    
TEST_DIR = os.path.join(DATA_DIR, "test")
TEST_PREDICT_DIR = os.path.join(DATA_DIR, "test_predicts")

#Input Dimensions
HEIGHT = 224
WIDTH = 224

output_shape = (HEIGHT, WIDTH)
input_shape = (HEIGHT, WIDTH)

N_CLASS = 3
