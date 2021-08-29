#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 19:51:13 2021

@author: recepaydogdu
"""
from constant import *
import glob
import tqdm 
import cv2 
from crop_sign import cropped_image

def visualize_sign(txt_path,save_box,cropped_path,cropped=False,save=False):
    global X_up, Y_up, X_bottom, Y_bottom
    txt_file=open(txt_path,"r")
    for line in tqdm.tqdm(txt_file):
        image_path=line.split(" ")[0]
        image=cv2.imread(image_path)
        for box in line.split(" ")[1:]:
            [X_up, Y_up, X_bottom, Y_bottom, classID] = box.split(',')
            result_box=cv2.rectangle(image, (int(X_up), int(Y_up)), (int(X_bottom),int(Y_bottom)),color=((38, 255, 255)), thickness=2)
            if save==True:
                cv2.imwrite(image_path.replace("images",save_box),result_box)
            if cropped==True:
                cropped_image(image_path, X_up,Y_up, X_bottom, Y_bottom,cropped_path)
    
txt_train_path="../../data/p2_data/train.txt"
save_box_train='result_train_box'
cropped_path_train='cropped_train_box'
visualize_sign(txt_train_path,save_box_train,cropped_path_train,cropped=True,save=True)


txt_valid_path="../../data/p2_data/valid.txt"
save_box_valid='result_valid_box'
cropped_path_valid='cropped_valid_box'
visualize_sign(txt_valid_path,save_box_valid,cropped_path_valid,cropped=True,save=True)