import cv2
import numpy as np
import os
from os.path import isfile, join
from natsort import natsorted
# Opens the Video file
cap = cv2.VideoCapture('/content/drive/MyDrive/Studies/test/results3.avi')
i = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    cv2.imwrite('frames/frame'+str(i)+'.jpg',frame)
    i+=1
 
cap.release()
cv2.destroyAllWindows()

#FRAME TO VIDEO
pathIn= '/content/data/test_predicts/Unet_2/'
pathOut = 'video24.avi'
fps = 24
frame_array = []
files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
#for sorting the file names properly
files = natsorted(files)
for i in range(len(files)):
    filename=pathIn + files[i]
    #reading each files
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    
    #inserting the frames into an image array
    frame_array.append(img)
out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])
out.release()