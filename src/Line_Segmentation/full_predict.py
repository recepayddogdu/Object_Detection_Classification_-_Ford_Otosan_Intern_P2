import os
import glob
import torch
import tqdm
import cv2
from tqdm import tqdm_notebook
from preprocess import tensorize_image
import numpy as np
from constant import *
from train import *

#### PARAMETERS #####
cuda = True
test = True
fs_model_name = "Unet_2.pt"
line_model_name = "SegNet.pt"
fs_model_path = os.path.join(MODELS_DIR, fs_model_name)
line_model_path = os.path.join(MODELS_DIR, line_model_name)
input_shape = input_shape
#####################

if test:
    if not os.path.exists(TEST_PREDICT_DIR): 
      os.mkdir(TEST_PREDICT_DIR)
    test_input_path_list = glob.glob(os.path.join(TEST_DIR, "*"))
    test_input_path_list.sort()
    predict_path = os.path.join(TEST_PREDICT_DIR, model_name.split(".")[0])
else:
    if not os.path.exists(PREDICT_DIR): 
      os.mkdir(PREDICT_DIR)
    predict_path = os.path.join(PREDICT_DIR, model_name.split(".")[0])

if not os.path.exists(predict_path): 
    os.mkdir(predict_path)

# LOAD MODEL
fs_model = torch.load(fs_model_path)
#Remember that you must call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference. 
#Failing to do this will yield inconsistent inference results.
fs_model.eval()

line_model = torch.load(line_model_path)
line_model.eval()

if cuda:
    fs_model = fs_model.cuda()
    line_model = line_model.cuda()



# PREDICT
def predict(fs_model, line_model, images):


    for image in tqdm_notebook(images):
        img = cv2.imread(image)
        batch_test = tensorize_image([image], input_shape, cuda)
        
        fs_output = fs_model(batch_test)
        line_output = line_model(batch_test)
        fs_out = torch.argmax(fs_output, axis=1)
        line_out = torch.argmax(line_output, axis=1)

        
        fs_out_cpu = fs_out.cpu()
        line_out_cpu = line_out.cpu()
        
        fs_outputs_list  = fs_out_cpu.detach().numpy()
        line_outputs_list  = line_out_cpu.detach().numpy()
        
        fs_mask = np.squeeze(fs_outputs_list, axis=0)
        line_mask = np.squeeze(line_outputs_list, axis=0)
       
        fs_mask_uint8 = fs_mask.astype('uint8')
        line_mask_uint8 = line_mask.astype('uint8')
        
        fs_mask_resize = cv2.resize(fs_mask_uint8, ((img.shape[1]), (img.shape[0])), interpolation = cv2.INTER_CUBIC)
        fs_line_resize = cv2.resize(line_mask_uint8, ((img.shape[1]), (img.shape[0])), interpolation = cv2.INTER_NEAREST)
        
        
        #img_resize = cv2.resize(img, input_shape)
        mask_ind = fs_mask_resize == 1
        mask_ind = fs_line_resize == 1
        #copy_img = img_resize.copy()
        copy_img = img.copy()
        
        img[fs_mask_resize==1, :] = (255, 0, 125)
        img[fs_line_resize==1, :] = (0, 0, 255)
        img[fs_line_resize==2, :] = (38, 255, 255)
        
        opac_image = (img/2 + copy_img/2).astype(np.uint8)
        cv2.imwrite(os.path.join(predict_path, image.split("/")[-1]), opac_image)
        #print("mask size from model: ", mask.shape),
        #print("resized mask size: ", mask_resize.shape)

if __name__ == "__main__":
    predict(fs_model, line_model, test_input_path_list)