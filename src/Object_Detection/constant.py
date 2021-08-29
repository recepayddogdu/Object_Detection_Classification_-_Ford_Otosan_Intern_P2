import os

DATA_DIR = "../../data/p2_data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")

JSON_DIR = os.path.join(DATA_DIR, "jsons") #Annotation dosyalarının dosya yolu.

MODELS_DIR = "../../models"

OD_TRA_LABEL='../../data/p2_data/train.txt'  
OD_TES_LABEL='../../data/p2_data/test.txt'  
OD_VAL_LABEL='../../data/p2_data/valid.txt'  
  
result_train_box ='../../data/p2_data/result_train_box'
result_valid_box ='../../data/p2_data/result_valid_box'
#If there is no file in the given file path, a new file is created
if not os.path.exists(result_valid_box): 
    os.mkdir(result_valid_box)

if not os.path.exists(result_train_box): 
    os.mkdir(result_train_box)
    
cropped_path_valid='../../data/p2_data/cropped_valid_box'
cropped_path_train ='../../data/p2_data/cropped_train_box'

if not os.path.exists(cropped_path_valid): 
    os.mkdir(cropped_path_valid)

if not os.path.exists(cropped_path_train): 
    os.mkdir(cropped_path_train)
    
train='../../data/p2_data/obj'
test='../../data/p2_data/test_pred'
valid='../../data/p2_data/test'

if not os.path.exists(train): 
    os.mkdir(train)

if not os.path.exists(test): 
    os.mkdir(test)
    
if not os.path.exists(valid): 
    os.mkdir(valid)    