#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 19:51:45 2021

@author: recepaydogdu
"""
from constant import *
import glob
import tqdm 
import cv2

def cropped_image(image_path, X_up,Y_up, X_bottom, Y_bottom,cropped_path):
    img = cv2.imread(image_path)
    w = int(X_bottom) - int(X_up)
    h = int(Y_bottom) -int(Y_up)
    crop_img = img[int(Y_up):int(Y_up)+h,int( X_up):int(X_up)+w]
    
    
    cv2.imwrite(image_path.replace('images',cropped_path),crop_img)