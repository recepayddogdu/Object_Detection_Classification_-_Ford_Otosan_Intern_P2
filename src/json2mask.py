#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 00:44:19 2021

@author: aycaburcu
"""
import os
import cv2
import tqdm
import json
import numpy as np
import matplotlib.pyplot as plt
from constant import *

jsons=os.listdir(JSON_DIR)#List created with names of json files in ann folder



for json_name in tqdm.tqdm(jsons):#access the elements in the json list
    json_path = os.path.join(JSON_DIR, json_name)#Merged json_dir with json_name and created file path
    json_file = open(json_path, 'r')#file reading process
    json_dict=json.load(json_file)#Contents of json file converted to dict data type
    mask_fre=np.zeros((json_dict["size"]["height"],json_dict["size"]["width"]), dtype=np.uint8)
    mask_line=np.zeros((json_dict["size"]["height"],json_dict["size"]["width"]), dtype=np.uint8)
    
    mask_path = os.path.join(MASK_DIR, json_name[:-5])
    # The values of the object keys in the dicts that we obtained from each 	json file have been added to the list.
    
    for obj in json_dict["objects"]:# To access each list inside the json_objs list
        
        
        if obj['classTitle']=='Freespace':#Objects whose classtitle is freespace
            
            cv2.fillPoly(mask_fre,np.array([obj['points']['exterior']],dtype=np.int32),color=100)
          
            if obj['points']['interior'] !=[]: #Checking if interior list is not empty
                for interior in obj['points']['interior']:
                         #Converting to int32 because there are floats between points
                         cv2.fillPoly(mask_fre,np.array([interior],dtype=np.int32),color=0)
                         
        if obj['classTitle']=='Solid Line':
           a=np.array([obj['points']['exterior']])
           cv2.polylines(mask_line,np.array([obj['points']['exterior']]),False,color=50,thickness=16)
        elif obj['classTitle']=='Dashed Line':
           cv2.polylines(mask_line,np.array([obj['points']['exterior']]),False,color=200,thickness=16)
           
       
            
    mask=cv2.bitwise_or(mask_fre,mask_line)
    mask[mask==236]=200
    mask[mask==118]=50

    cv2.imwrite(mask_path,mask.astype(np.uint8))#Print filled masks in mask_path with imwrite