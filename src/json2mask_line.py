import os
import cv2
import tqdm
import json
import numpy as np
from constant import *

if not os.path.exists(MASK_LINE_DIR): #MASK_DIR yolunda masks klasörü yoksa yeni klasör oluştur.
    os.mkdir(MASK_LINE_DIR)

jsons=os.listdir(JSON_DIR)#List created with names of json files in ann folder

for json_name in tqdm.tqdm(jsons):#access the elements in the json list
    json_path = os.path.join(JSON_DIR, json_name)#Merged json_dir with json_name and created file path
    json_file = open(json_path, 'r')#file reading process
    json_dict=json.load(json_file)#Contents of json file converted to dict data type
    mask=np.zeros((json_dict["size"]["height"],json_dict["size"]["width"]), dtype=np.uint8)
   
    mask_path = os.path.join(MASK_LINE_DIR, json_name[:-5])
    # The values of the object keys in the dicts that we obtained from each 	json file have been added to the list.
    
    for obj in json_dict["objects"]:# To access each list inside the json_objs list                   
        if obj['classTitle']=='Solid Line':
           cv2.polylines(mask,np.array([obj['points']['exterior']],dtype=np.int32),False,color=1,thickness=14)
 
        elif obj['classTitle']=='Dashed Line':       
               cv2.polylines(mask,np.array([obj['points']['exterior']],dtype=np.int32),False,color=2,thickness=9)
    
    cv2.imwrite(mask_path,mask.astype(np.uint8))#Print filled masks in mask_path with imwrite