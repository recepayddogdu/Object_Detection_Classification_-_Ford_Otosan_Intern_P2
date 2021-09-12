import os
import cv2
import numpy as np
import tqdm
from constant import *

if not os.path.exists(MASKED_IMAGES_DIR):
    os.mkdir(MASKED_IMAGES_DIR)

mask_name_polygon = os.listdir(MASK_POLY_DIR)
#mask_name_line = os.listdir(MASK_LINE_DIR)        
        
for mask_name in tqdm.tqdm(mask_name_polygon):
    img_name = mask_name[:-3]+"jpg" #images dosyasında isim karşılığını almak için uzantısını .jpg alıyoruz
    
    image = cv2.imread(os.path.join(IMAGE_DIR, img_name)).astype(np.uint8)
    
    mask_polygon = cv2.imread(os.path.join(MASK_POLY_DIR, mask_name), 0).astype(np.uint8)
    mask_line = cv2.imread(os.path.join(MASK_LINE_DIR, mask_name), 0).astype(np.uint8)
    #0-255 aralığına almak için uint8 yapıyoruz.
    
    cpy_image = image.copy() #orjinal image'ın kopyası
    
    image[mask_polygon==100, :] = (255, 0, 125) #color=100 olan mask'ların konumlarını image'da renklendiriyoruz
    image[mask_line==1, :] = (0, 0, 255)
    image[mask_line==2, :] = (38, 255, 255)
    
    opac_image = (image/2 + cpy_image/2).astype(np.uint8)
    
    
    cv2.imwrite(os.path.join(MASKED_IMAGES_DIR, mask_name), opac_image)
