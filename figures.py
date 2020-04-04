import cv2
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import random
import time
import os
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure





path1 = 'Individual_Component/train/train_positive_A/00000000/'
file_name = 'item_00000000.pnm'
img = cv2.imread(path1+file_name)
resized_img = cv2.resize(img, (64, 128))

fig, (ax1) = plt.subplots(1, 1, figsize=(16, 8)) 
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cmap=plt.cm.gray) 
ax1.set_title('Input image')
plt.show()

fig, (ax1) = plt.subplots(1, 1, figsize=(16, 8)) 
ax1.imshow(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB), cmap=plt.cm.gray) 
ax1.set_title('Resized image : 64 x 128') 
plt.show()
    

_, hog_image1 = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), visualize=True, multichannel=True)    

_, hog_image2 = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(1, 1), visualize=True, multichannel=True)
     
hog_image_rescaled1 = exposure.rescale_intensity(hog_image1, in_range=(0, 10)) 
hog_image_rescaled2 = exposure.rescale_intensity(hog_image2, in_range=(0, 10))
    
fig, (ax1) = plt.subplots(1, 1, figsize=(16, 8)) 
ax1.imshow(hog_image_rescaled1, cmap=plt.cm.gray) 
ax1.set_title('Histogram of Oriented Gradients : 2x2 cells per block')
plt.show()

fig, (ax1) = plt.subplots(1, 1, figsize=(16, 8))        
ax1.imshow(hog_image_rescaled2, cmap=plt.cm.gray) 
ax1.set_title('Histogram of Oriented Gradients : 1x1 cells per block')
plt.show()



path2 = 'Individual_Component/train/train_negative_A/00000000/'
img = cv2.imread(path2+file_name)
resized_img = cv2.resize(img, (64, 128))

fig, (ax1) = plt.subplots(1, 1, figsize=(16, 8)) 
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cmap=plt.cm.gray) 
ax1.set_title('Input image')
plt.show()

fig, (ax1) = plt.subplots(1, 1, figsize=(16, 8)) 
ax1.imshow(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB), cmap=plt.cm.gray) 
ax1.set_title('Resized image : 64 x 128') 
plt.show()
    

_, hog_image1 = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), visualize=True, multichannel=True)    

_, hog_image2 = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(1, 1), visualize=True, multichannel=True)
     
hog_image_rescaled1 = exposure.rescale_intensity(hog_image1, in_range=(0, 10)) 
hog_image_rescaled2 = exposure.rescale_intensity(hog_image2, in_range=(0, 10))
    
fig, (ax1) = plt.subplots(1, 1, figsize=(16, 8)) 
ax1.imshow(hog_image_rescaled1, cmap=plt.cm.gray) 
ax1.set_title('Histogram of Oriented Gradients : 2x2 cells per block')
plt.show()

fig, (ax1) = plt.subplots(1, 1, figsize=(16, 8))        
ax1.imshow(hog_image_rescaled2, cmap=plt.cm.gray) 
ax1.set_title('Histogram of Oriented Gradients : 1x1 cells per block')
plt.show()
