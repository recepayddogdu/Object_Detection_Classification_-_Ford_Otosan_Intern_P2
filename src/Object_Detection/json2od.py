#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 17:29:27 2021

@author: recepaydogdu
"""
import os
import cv2
import tqdm
import json
import numpy as np
import matplotlib.pyplot as plt
from constant import *

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
                
                # Add into list
                annotations.append(str(obj['points']['exterior'][0][0]) + ',' + str(obj['points']['exterior'][0][1]) + ',' + str(obj['points']['exterior'][1][0]) + ',' + str(obj['points']['exterior'][1][1]) + ',' + str(obj_id))
                
     # Modify list
    strlabel = ''
    for idx in range(len(annotations)):
        if idx != 0:
            strlabel += ' '

        strlabel += annotations[idx]

    return strlabel

""" Write down into the txt file """

def txt_write(jsons,valid_size,test_size,train_label,test_label,valid_label):
    lines= []
    for json_name in tqdm.tqdm(jsons):
        
        image_name = os.path.splitext(json_name)[0]
        # Change from png to jpg
        image_name=image_name[:-3]+'jpg'
        image_path = os.path.join(IMAGE_DIR, image_name)
    
        if len(json2od(JSON_DIR,json_name))!=0:
            line = image_path+' '+json2od(JSON_DIR,json_name)+'\n'
            lines.append(line)
    
    total_size=len(lines)
    test_ind  = int(total_size * test_size)#Multiply indices length by test_size and assign it to an int-shaped variable
    valid_ind = int(test_ind + total_size * valid_size)
    train_txt=open(train_label, "w+")
    test_txt=open(test_label, "w+")
    valid_txt=open(valid_label, "w+")
    
    for ln in lines[:test_ind] :
        # Write down
        test_txt.write(ln)
            
    for ln in lines[test_ind:valid_ind]:
        valid_txt.write(ln)
          
    for ln in lines[valid_ind:]:
        train_txt.write(ln)
      
 
        

jsons=os.listdir(JSON_DIR)#List created with names of json files in ann folde
valid_size = 0.3#Validation dataset is used to evaluate a particular model, but this is for frequent evaluation.
test_size  = 0.05#rate of data to be tested
train_label=OD_TRA_LABEL
test_label=OD_TES_LABEL
valid_label=OD_VAL_LABEL
txt_write(jsons,valid_size,test_size,train_label,test_label,valid_label)   