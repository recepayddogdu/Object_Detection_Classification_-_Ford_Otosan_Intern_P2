#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 19:22:17 2021

@author: recepaydogdu
"""
import os
import cv2
import tqdm
import json
import numpy as np
import matplotlib.pyplot as plt
from constant import *
from shutil import copyfile

def json2od(JSON_DIR,json_name):
    
    json_path = os.path.join(JSON_DIR, json_name)#Merged json_dir with json_name and created file path
    json_file = open(json_path, 'r')#file reading process
    json_dict=json.load(json_file)#Contents of json file converted to dict data type
    
       
    if len(json_dict["objects"]) == 0:
            a='Object not found!'
            return a
    obj_id=0
    annotations = []
    for obj in json_dict["objects"]:# To access each list inside the json_objs list
        
        if obj['classTitle']=='Traffic Sign':
                  
                # Eliminate small ones 
                if (obj['points']['exterior'][1][0] - obj['points']['exterior'][0][0]) < 16 or (obj['points']['exterior'][1][1] - obj['points']['exterior'][0][1]) < 16:
                    continue
                
                xmin=obj['points']['exterior'][0][0]
                ymin=obj['points']['exterior'][0][1]
                xmax=obj['points']['exterior'][1][0]
                ymax=obj['points']['exterior'][1][1]
                                
                def width():
                    width=int(xmax-xmin)
                    return width
                def height():
                    height=int(ymax-ymin)
                    return height
                def x_center():
                    x_center=int(xmin + width()/2)
                    return x_center
                def y_center():
                    y_center=int(ymin + height()/2)
                    return y_center
                annotations.append(str(obj_id)+" "+str(x_center()/1920)+" "+str(y_center()/1208)+" "+str(width()/1920)+" "+str(height()/1208))
                
     # Modify list
    strlabel = ''
    for idx in range(len(annotations)):
        if idx != 0:
            strlabel += '*'

        strlabel += annotations[idx]

    return strlabel

""" Write down into the txt file """

def txt_write(jsons,valid_size,test_size):
    global lines
    global split_data
    lines= []
    
    for json_name in tqdm.tqdm(jsons):
        
        image_name = os.path.splitext(json_name)[0]
        # Change from png to jpg
        image_name=image_name[:-3]+'jpg'
        image_path = os.path.join(IMAGE_DIR, image_name)
    
        if len(json2od(JSON_DIR,json_name))!=0:
            line = image_path+'*'+json2od(JSON_DIR,json_name)
            lines.append(line)
    
    total_size=len(lines)
    test_ind  = int(total_size * test_size)#Multiply indices length by test_size and assign it to an int-shaped variable
    valid_ind = int(test_ind + total_size * valid_size)

    
    for ln in lines[:test_ind] :
        split_data=ln.split("*")
        image_path=split_data[0]
        copy_image=split_data[0].replace("images","test_pred")
        copyfile(image_path,copy_image)

            
    for ln in lines[test_ind:valid_ind]:
        split_data=ln.split("*")
        image_path=split_data[0]
        copy_image=split_data[0].replace("images","test")
        copyfile(image_path,copy_image)
        txt_path=(split_data[0][:-3]+"txt").replace("images","test")
        with open(txt_path, 'w') as f:
            for data in split_data[1:]:
                f.write("%s\n" % data)
            f.close()
        
          
    for ln in lines[valid_ind:]:
        split_data=ln.split("*")
        image_path=split_data[0]
        copy_image=split_data[0].replace("images","obj")
        copyfile(image_path,copy_image)
        txt_path=(split_data[0][:-3]+"txt").replace("images","obj")
        with open(txt_path, 'w') as f:
            for data in split_data[1:]:
                f.write("%s\n" % data)
            f.close()
      
 
jsons=os.listdir(JSON_DIR)#List created with names of json files in ann folde
valid_size = 0.2#Validation dataset is used to evaluate a particular model, but this is for frequent evaluation.
test_size  = 0.05#rate of data to be tested

txt_write(jsons,valid_size,test_size)       