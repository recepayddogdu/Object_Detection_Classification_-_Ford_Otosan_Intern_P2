from Unet_1 import FoInternNet
from preprocess import tensorize_image, tensorize_mask, image_mask_check
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from constant import *
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
from torchsummary import summary
from tqdm import tqdm_notebook

######### PARAMETERS ##########
valid_size = 0.3 # Rate of validation dataset
test_size  = 0.1 # Rate of test dataset
batch_size = 4   # Number of data to be processed simultaneously in the model
epochs = 25      # Epoch count is the number of times all training data is shown to the network during training.
cuda = True
input_shape = input_shape
n_classes = 2
augmentation = True
checkpoint = False
model_name = "Unet_2-augmentation"
cp_epoch = 0
###############################
    
# PREPARE IMAGE AND MASK LISTS
image_path_list = glob.glob(os.path.join(IMAGE_DIR, "*"))
image_path_list.sort()
# The names of the files in the IMAGE_DIR path are listed and sorted
mask_path_list = glob.glob(os.path.join(MASK_DIR, "*"))
mask_path_list.sort()

# DATA CHECK
image_mask_check(image_path_list, mask_path_list)
"""
    Since it is supervised learning, there must be an expected output for each
    input. This function assumes input and expected output images with the
    same name.
"""

# SHUFFLE INDICES
indices = np.random.permutation(len(image_path_list)) # create a random ordered array of indices

# DEFINE TEST AND VALIDATION INDICES
# Multiply indices length by test_size and assign it to an int-shaped variable
test_ind = int(len(indices) * test_size) #test_size = 0.1 = 855 
valid_ind = int(len(indices) * valid_size) #valid_size = 0.2 = 1711 

# SLICE TEST DATASET FROM THE WHOLE DATASET
test_input_path_list = image_path_list[:test_ind] #Get 0 to 855 elements of the image_path_list = 855 elements
test_label_path_list = mask_path_list[:test_ind]  #Get 0 to 855 elements of the mask_path_list = 855 elements

# SLICE VALID DATASET FROM THE WHOLE DATASET
valid_input_path_list = image_path_list[test_ind:valid_ind] #Get 855 to 1711 elements of the image_path_list = 856 elements
valid_label_path_list = mask_path_list[test_ind:valid_ind]  #Get 855 to 1711 elements of the mask_path_list = 856 elements

# SLICE TRAIN DATASET FROM THE WHOLE DATASET
train_input_path_list = image_path_list[valid_ind:] #Get the elements of the image_path_list from 1711 to the last element = 6844 elements
train_label_path_list = mask_path_list[valid_ind:]  #Get the elements of the mask_path_list from 1711 to the last element = 6844 elements

if augmentation:
    # PREPARE AUGMENTATED IMAGE AND MASK LISTS
    aug_image_list = glob.glob(os.path.join(AUG_IMAGE_DIR, "*"))
    aug_image_list.sort()
    aug_mask_list = glob.glob(os.path.join(AUG_MASK_DIR, "*"))
    aug_mask_list.sort()
    
    aug_size = int(len(aug_mask_list)/2)
    
    train_input_path_list = aug_image_list[:aug_size] + train_input_path_list + aug_image_list[aug_size:]
    train_label_path_list = aug_mask_list[:aug_size] + train_label_path_list + aug_mask_list[aug_size:]

def save_checkpoint(epoch, model, optimizer, val_loss, model_name):
  if not os.path.exists(MODELS_DIR):
    os.mkdir(MODELS_DIR)
  model_name += "-checkpoint-"+str(epoch)+".pt"
  
  torch.save({
              "epoch": epoch,
              "model_state_dict": model.state_dict(),
              "optimizer_state_dict": optimizer.state_dict(),
              #"loss": val_loss                  
              }, os.path.join(MODELS_DIR, model_name))
  os.system("zip -r models.zip models")
  os.system("cp models.zip /content/drive/MyDrive/InternP1")
  print("Checkpoint Saved!")

def load_checkpoint(model, optimizer, model_name, cp_epoch):
  checkpoint = torch.load(os.path.join(MODELS_DIR, model_name+"-checkpoint-"+str(cp_epoch)+".pt"))
  model.load_state_dict(checkpoint["model_state_dict"])
  optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
  #epoch_cp = checkpoint["epoch"]

if __name__ == "__main__":
    
    # DEFINE STEPS PER EPOCH
    steps_per_epoch = len(train_input_path_list)//batch_size
    
    # CALL MODEL
    model = FoInternNet(input_size=input_shape, n_classes=n_classes)
    
    # DEFINE LOSS FUNCTION AND OPTIMIZER
    criterion = nn.BCELoss() #Creates a criterion that measures the Binary Cross Entropy between target and output:
    optimizer = optim.Adam(model.parameters(), lr=0.001) #Commonly used momentum beta coefficient is 0.9
    # lr = Learning Rate
    
    # IF CUDA IS USED, IMPORT THE MODEL INTO CUDA
    if cuda:
        model = model.cuda()
    
    
    val_losses=[]
    train_losses=[]

    if checkpoint:
      load_checkpoint(model, optimizer, model_name, cp_epoch)
      epochs = epochs - (cp_epoch+1)
      val_losses = val_losses_cp
      train_losses = train_losses_cp

    # TRAINING THE NEURAL NETWORK
    for epoch in tqdm_notebook(range(epochs)):
    
        running_loss = 0
        #In each epoch, images and masks are mixed randomly in order not to output images sequentially.
        pair_IM=list(zip(train_input_path_list,train_label_path_list))
        np.random.shuffle(pair_IM)
        unzipped_object=zip(*pair_IM)
        zipped_list=list(unzipped_object)
        train_input_path_list=list(zipped_list[0])
        train_label_path_list=list(zipped_list[1])
    
        for ind in tqdm_notebook(range(steps_per_epoch), leave=False):
            batch_input_path_list = train_input_path_list[batch_size*ind:batch_size*(ind+1)]
            #train_input_path_list [0: 4] gets first 4 elements on first entry
            #in the second loop train_input_list [4: 8] gets the second 4 elements
            #element advances each time until batch_size
            batch_label_path_list = train_label_path_list[batch_size*ind:batch_size*(ind+1)]
            batch_input = tensorize_image(batch_input_path_list, input_shape, cuda)
            batch_label = tensorize_mask(batch_label_path_list, input_shape, n_classes, cuda)
    
            optimizer.zero_grad() #resets the gradian otherwise accumulation occurs on each iteration
    
            outputs = model(batch_input)
            
            # Forward passes the input data
            loss = criterion(outputs, batch_label)
            loss.backward() # Calculates the gradient, how much each parameter needs to be updated
            optimizer.step() # Updates each parameter according to the gradient
            
            running_loss += loss.item() # loss.item() takes the scalar value held in loss.
            #print(ind)
            
            ### VALIDATION ###
            if ind == steps_per_epoch-1:
                train_losses.append(running_loss)
                print('training loss on epoch {}: {}'.format(epoch, running_loss))
                
                val_loss = 0
                for (valid_input_path, valid_label_path) in zip(valid_input_path_list, valid_label_path_list):
                    batch_input = tensorize_image([valid_input_path], input_shape, cuda)
                    batch_label = tensorize_mask([valid_label_path], input_shape, n_classes, cuda)
                    outputs = model(batch_input)
                    loss = criterion(outputs, batch_label)
                    val_loss += loss
                    val_losses.append(val_loss)
                    break
    
                print('validation loss on epoch {}: {}\n'.format(epoch, val_loss))

                save_checkpoint(epoch, model, optimizer, val_loss, model_name)

    def save_model(model, model_name, val_losses, train_losses, epochs):
        if not os.path.exists(MODELS_DIR):
            os.mkdir(MODELS_DIR)
        model_name += ".pt"
        torch.save(model, os.path.join(MODELS_DIR, model_name))
        print("Model Saved!")
        
        # DRAW GRAPH
        norm_validation = [float(i)/sum(val_losses) for i in val_losses]
        norm_train = [float(i)/sum(train_losses) for i in train_losses]
        #norm_validation = val_losses
        #norm_train = train_losses
        epoch_numbers=list(range(1,epochs,1))
        plt.figure(figsize=(12,6))
        plt.subplot(2, 2, 1)
        plt.plot(epoch_numbers,norm_validation,color="red") 
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
        plt.title('Train losses')
        plt.subplot(2, 2, 2)
        plt.plot(epoch_numbers,norm_train,color="blue")
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
        plt.title('Validation losses')
        plt.subplot(2, 1, 2)
        plt.plot(epoch_numbers,norm_validation, 'r-',color="red")
        plt.plot(epoch_numbers,norm_train, 'r-',color="blue")
        plt.legend(['w=1','w=2'])
        plt.title('Train and Validation Losses')
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
        plt.tight_layout()
        plt.savefig(os.path.join(MODELS_DIR, model_name.split(".")[0]+"-loss.png"))
        plt.show()

    save_model(model, model_name, val_losses, train_losses, epochs)

    #summary(model, input_shape)