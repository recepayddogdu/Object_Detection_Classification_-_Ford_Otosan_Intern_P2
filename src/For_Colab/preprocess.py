import glob
import cv2
import torch
import numpy as np
from constant import *

def tensorize_image(image_path_list, output_shape, cuda=False):
    batch_images = [] # Create empty list
    global img
    for image_path in image_path_list: # For each image
    
        img = cv2.imread(image_path) # Access and read image
        
        zeros_img = np.zeros((1920, 1208))
        norm_img = cv2.normalize(img, zeros_img, 0, 255, cv2.NORM_MINMAX)
        
        img = cv2.resize(norm_img, output_shape, interpolation = cv2.INTER_NEAREST) # Resize the image according to defined shape
        
        # Change input structure according to pytorch input structure
        torchlike_image = torchlike_data(img)
        
        batch_images.append(torchlike_image) # Add into the list
    
    # Convert from list structure to torch tensor
    image_array = np.array(batch_images, dtype=np.float32)
    torch_image = torch.from_numpy(image_array).float()
    # The tensor should be in [batch_size, output_shape[0], output_shape[1], 3] shape.
    
    # If multiprocessing is chosen
    if cuda:
        torch_image = torch_image.cuda()
        
    return torch_image



def tensorize_mask(mask_path_list, output_shape, N_CLASS, cuda=False):
    batch_masks = []
    global mask
    global resize_mask
    global mask_path
    
    for mask_path in mask_path_list:
        
        # Access and read mask
        mask = cv2.imread(mask_path, 0)
        
        # Resize the image according to defined shape
        mask = cv2.resize(mask, output_shape, interpolation = cv2.INTER_NEAREST)
        resize_mask = mask
        
        
        #Apply One-Hot Encoding to image
        mask = one_hot_encoder(mask, N_CLASS)
        
        # Change input structure according to pytorch input structure
        torchlike_mask = torchlike_data(mask)
        
        batch_masks.append(torchlike_mask)
    
    mask_array = np.array(batch_masks, dtype=np.int)
    torch_mask = torch.from_numpy(mask_array).float()
    
    if cuda:
        torch_mask = torch_mask.cuda()
    
    return torch_mask



def image_mask_check(image_path_list, mask_path_list):
    # Check list lenghts
    if len(image_path_list) != len(mask_path_list):
        print("There are missing files! Images and masks folder should have same number of files.")
        return False
    
    # Check each file names
    for image_path, mask_path in zip(image_path_list, mask_path_list):
        image_name = image_path.split("/")[-1].split(".")[0]
        mask_name = mask_path.split("/")[-1].split(".")[0]
        
        if image_name != mask_name:
            print("Image and mask name does no match {} - {}".format(image_name, mask_name)
                  + "\nImages and masks folder should have same file names.")
            return False
        
    return True
    


def torchlike_data(data):
        
    #transpose process 
    n_channels = data.shape[2]
    torchlike_data = np.empty((n_channels, data.shape[0], data.shape[1]))#Returns a new array of the given shape and type.
    #creates an array of these sizes
    for ch in range(n_channels):# generates ch numbers as long as the list
        torchlike_data[ch] = data[:,:,ch] #torchlike_data[0]=data[:,:,0] 
        #Export data in data individually to torchlike_data
    return torchlike_data


def one_hot_encoder(data, n_class):
    
    #one hot encode
    #Create an np.array of zeros.
    one_hot=np.zeros((data.shape[0],data.shape[1],n_class),dtype=np.int)
    #Find unique values in res_mask [0,1]
    #increase in i by the length of the list
    #[0,1] when returning the inside of list, each list element is given to unique_value variable
    global unique_values
    unique_values = np.unique(data)
    
    for i,unique_value in enumerate(np.unique(data)):
        one_hot[:,:,i][data==unique_value]=1
    return one_hot


if __name__=="__main__":
    
    
    # Access images
    image_list = glob.glob(os.path.join(IMAGE_DIR, '*'))
    image_list.sort()

    # Access masks
    mask_list = glob.glob(os.path.join(MASK_DIR, '*'))
    mask_list.sort()
    
    # Check image-mask match
    if image_mask_check(image_list, mask_list):
        
        
        # Take image to number of batch size
        batch_image_list = image_list[:BATCH_SIZE]
        global batch_image_tensor
        # Convert into torch tensor
        batch_image_tensor = tensorize_image(batch_image_list, output_shape)
        
        # Check
        print("For features:\ndtype is " + str(batch_image_tensor.dtype))
        print("Type is " + str(type(batch_image_tensor)))
        print("The size should be [" + str(BATCH_SIZE) + ", 3, " + str(HEIGHT) + ", " + str(WIDTH) + "]")
        print("Size is " + str(batch_image_tensor.shape)+"\n")
        
        # Take masks to number of batch size
        batch_mask_list = mask_list[:BATCH_SIZE]
        
        # Convert into torch tensor
        batch_mask_tensor = tensorize_mask(batch_mask_list, output_shape, 2)
        
        # Check
        print("For labels:\ndtype is "+str(batch_mask_tensor.dtype))
        print("Type is "+str(type(batch_mask_tensor)))
        print("The size should be ["+str(BATCH_SIZE)+", 2, "+str(HEIGHT)+", "+str(WIDTH)+"]")
        print("Size is "+str(batch_mask_tensor.shape))
