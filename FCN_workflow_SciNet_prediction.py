#!/usr/bin/env python

################################################
### THIS IS THE TRAINING/VALIDATION PIPELINE ###
################################################

#Code adapted by Jeff Neyhart from code developed by Pranav Raja, Devin Rippner, and Alexander Buchko

# # ***Welcome to the Training and Inference Pipeline ***
# 
# # ***Step 1: Mount google drive***
# 
# # ***Input code from google to access your drive***
# 
# *** Consider running your instance locally. This will require modification of the file name paths, but will allow use on computers with more resources.***
# 
# ### *To run a local instance:*
# 
# jupyter notebook  --NotebookApp.allow_origin='https://colab.research.google.com'  --port=8080  --NotebookApp.port_retries=0


# #**Materials**
#   Input the material mask name and information below.
# 
#   Specifically:
#  
#   **name** - The name for the material. This is pretty arbitrary, but it will be
#   used to label output folders and images.
#  
#   **input_rbg_vals** - The rbg values of the material in the input mask image.
#  
#   **output_val** - The greyscale value of the mask when you output the images.
#   This is arbitrary, but every material should have its own output color
#   so they can be differentiated
#  
#   **confidence_threshold** - The lower this number, the more voxels will be labled a specific material. Essentially, the ML algorith outptus a confdience value  (centered on 0.5) for every voxel and every material. By default, voxels with  a confidence of 0.5 or greater are determined to be the material in question.  But we can labled voxles with a lower condience level by changing this  parameter
#   
#   **training_image_directory /training_mask_directory**: Input the directory where your training images and masks are located.
# 
#   **validation_fraction**: Input the fraction of images you want to validate your model during training. These are not a independent validation, but are part of the training process.
# 
#   **num_models**: Enter the number of models you want to iteratively train. Because these are statistical models, the performance of any given model will vary. Training more models will allow you to select the model that best fits your data.
#   
#   **num_epochs**: Enter number of epochs that you want to use to train your model. More is generally better, but takes more time.
# 
#   **batch_size**: Input your batch size. Larger batch sizes allow for faster training, but take up more VRAM. If you are running out of VRAM during training, decrease your batch size.
# 
#   **scale**: Input how you want your images scaled during model training and inference. When the scale is 1, your images will be used at full size for training. When the scale is less than 1, your images will be downsized according to the scale you set for training and inference, decreasing VRAM usage. If you run out of VRAM during training, consider rescaling your images.
# 
#   **models_directory**: Directory where your models are saved.
# 
#   **model_group**: Name for the group models you iteratively generate.
# 
#   **current_model_name**: Name for each individual model you generate; will automatically be labeled 1 through n for the number of models you specify above.
# 
#   **val_images/val_masks**: Input the directory where your independent validation images and masks are located. These images are not used for training and are used as an independent validation of your model.
# 
#   **csv_directory**: Directory where a CSV file of your validation results will be saved.
# 
#   **inference_directory**: Directory where the images you want analyzed are located.
# 
#   **output_directory**: Directory where you want your analysis results to be saved.
# 
# 

# In[ ]

# Import datetime for saving the date
from datetime import datetime
# Create a date string
ds = datetime.now().strftime("%Y%m%d-%H%M%S")


class Material:
 
  def __init__(self, name, input_rgb_vals, output_val, confidence_threshold=0):
    self.name = name
    self.input_rgb_vals = input_rgb_vals
    self.output_val = output_val
    self.confidence_threshold = confidence_threshold

#Creating a list of materials so we can iterate through it
materials = [
             Material("notberry", [0,0,0], 1, 0.5),
             Material("berry", [255,255,255], 100, 0.5),
             ]

# Boolean whether to retrain models or use current models
new_training = False

# Project directory
# IMPORTANT - ALL DIRECTORIES NEED TO END IN A /
proj_dir = "/path/to/project/directory/"
 
num_materials =len(materials)

#Various input/output directories
# IMPORTANT: END EACH DIRECTORY PATH WITH A "/"
training_image_directory = proj_dir + "train/images/"
training_mask_directory = proj_dir + "train/masks/"

#Fraction of total annotations you want to leave for validating the model.
validation_fraction=0.2

#Model Performance varies, make multiple models to have the best chance at success.
num_models=7

#Model Performance improves with increasing epochs, to a point.
num_epochs=70

# """Increasing batch size increase model training speed, but also eats up VRAM on the GPU. Find a balance between scale and batch size
# that best suits your needs"""
batch_size=3

#Decrease scale to decrease VRAM usage; if you run out of VRAM during traing, restart your runtime and down scale your images
scale=0.8

#Input model directory
# IMPORTANT: END EACH DIRECTORY PATH WITH A "/"
models_directory = proj_dir + "best_models/"

#Input the name you want to use for your group of models
model_group='model_group_name/'

## THIS IS NOT A DIRECTORY; DO NOT ADD TRAILING "/"
# Model name is based on the current date; so multiple runs are not clobbered
current_model_name = "model_name_" + ds + "_model"

# """Hold images/annotations in reserve to test your model performance. Use this metric to decide which model you want to use 
# for your data analysis"""
# IMPORTANT: END EACH DIRECTORY PATH WITH A "/"
test_images = proj_dir + "test/images/"
test_masks= proj_dir + "test/masks/"
csv_directory =  proj_dir + model_group.replace("/", "") + ".csv"

#Input the directory of the data you want to segment here.
inference_directory= proj_dir + 'inferenceImages/'

#Input the 5 alpha-numeric characters proceding the file number of your images
  #EX. Jmic3111_S0_GRID image_0.tif ----->mage_
proceeding="mage_"
#Input the 4 or mor alpha-numeric characters following the file number
  #EX. Jmic3111_S0_GRID image_0.tif ----->.tif
following=".jpg"

output_directory = proj_dir + model_group + 'watershed_adj/'


## Write variables to a file ##

# # List all objects
# objects = locals()
# 
# ## Write all of the above parameters to a python script that will be imported at later stages
# param_filename = proj_dir + "fcn_workflow_parameters_" + ds + ".py"
#
#
# # Open a file
# handle = open(param_filename, "w")
#
# # Write all the parameters to this file
# for key in objects:
#  value = objects[key]
#  # If the object value is one of the following, print it
#  if (type(value).__name__ in ["str", "int", "list"]):
#    if type(value).__name__ == "int":
#      handle.write(key + ' = ' + str(value) + '\n')
#    else:
#      handle.write(key + ' = "' + str(value) + '"\n')
#
# # Close the file
# handle.close()


################################################################################
################################################################################
### DO NOT EDIT BELOW THIS LINE ################################################
################################################################################
################################################################################






# pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html    


# #**Parameter Loading**

#Code Box 2
from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import random
#import scipy.ndimage as ndi
import albumentations as A
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from scipy.ndimage import morphology
from torch.utils.data import DataLoader, random_split
 
class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=scale, transform=False):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.transform=transform
        self.t_list=A.Compose([A.HorizontalFlip(p=0.4),A.VerticalFlip(p=0.4), A.Rotate(limit=(-50, 50), p=0.4),])
        self.means=[0]
        self.stds=[1]

        
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
 
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')
 
    def __len__(self):
        return len(self.ids)
 
 
    @classmethod
    def mask_preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))
 
        img_nd = np.array(pil_img)
 
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
 
       
        return img_nd
    
 
        
    def img_preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))
 
        img_nd = np.array(pil_img)
 
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
 
       
 
        return img_nd
 
    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '*')
        img_file = glob(self.imgs_dir + idx + '*')
 
        assert len(mask_file) == 1,             f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1,             f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])
 
  
        
 
        
        #Reshapes from 1 channel to 3 channels in grayscale
        img = self.img_preprocess(img, self.scale)
        mask = self.mask_preprocess(mask, self.scale)
        new_image=np.zeros((img.shape[0],img.shape[1],3))
        new_image[:,:,0]=img[:,:,0]
        new_image[:,:,1]=img[:,:,0]
        new_image[:,:,2]=img[:,:,0]
        
 
 
 
        img=new_image
 
        new_mask = np.zeros((num_materials,img.shape[0],img.shape[1]))
        # print(mask.shape)       
        for i, mat in enumerate(materials):
          # plt.imshow(mask[:,:,0])
          # plt.show()
          indices = np.all(mask == mat.input_rgb_vals, axis=-1)
          new_mask[i,:,:][indices] = 1
 
        mask = new_mask
  
        # plt.imshow(mask[1,:,:])
        # i=6
        # for i in range(len(mask)):
        #   plt.imshow(mask[i,:,:])
        #   plt.show()
        
        if img.max() > 1:
            img = img / 255
 
       
 
        
        if self.transform:
            augmented=self.t_list(image=img, masks=mask)
            img=augmented["image"]
            mask=augmented["masks"]
            
 
        
 
        img = img.transpose((2, 0, 1))
        
        mask=np.array(mask)
        
        
 
        
 
        img=torch.from_numpy(img)
        mask=torch.from_numpy(mask)
        
        img=transforms.Normalize(mean=self.means, std=self.stds)(img)
        return img, mask
        
        
dataset = BasicDataset(training_image_directory, training_mask_directory, scale=scale, transform=False)
 
#!!!!!!!!!!!!!!!!!!!!!!!!!!Set batch size here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# train, val=trainval_split(dataset, val_fraction=0.5)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)#, collate_fn=pad_collate)
#val_loader = DataLoader(val, batch_size=3, shuffle=False, num_workers=0, pin_memory=True)#, collate_fn=pad_collate)
nimages = 0
mean = 0.
std = 0.
for batch, _ in train_loader:
    # Rearrange batch to be the shape of [B, C, W * H]
    batch = batch.view(batch.size(0), batch.size(1), -1)
    # Update total number of images
    nimages += batch.size(0)
    # Compute mean and std here
    mean += batch.mean(2).sum(0) 
    std += batch.std(2).sum(0)
 
# Final step
mean /= nimages
std /= nimages
 
print(mean)
print(std)

dataset.means=mean
dataset.stds=std 

nimages = 0
newmean = 0.
newstd = 0.
for batch, _ in train_loader:
    # Rearrange batch to be the shape of [B, C, W * H]
    batch = batch.view(batch.size(0), batch.size(1), -1)
    # Update total number of images
    nimages += batch.size(0)
    # Compute mean and std here
    newmean += batch.mean(2).sum(0) 
    newstd += batch.std(2).sum(0)
 
# Final step
newmean /= nimages
newstd /= nimages
 
#!/usr/bin/env python


#######################################
### THIS IS THE PREDICTION PIPELINE ###
#######################################


class Material:
 
  def __init__(self, name, input_rgb_vals, output_val, confidence_threshold=0):
    self.name = name
    self.input_rgb_vals = input_rgb_vals
    self.output_val = output_val
    self.confidence_threshold = confidence_threshold
    
    
    

# #**Materials**
#   Input the material mask name and information below.
# 
#   Specifically:
#  
#   **name** - The name for the material. This is pretty arbitrary, but it will be
#   used to label output folders and images.
#  
#   **input_rbg_vals** - The rbg values of the material in the input mask image.
#  
#   **output_val** - The greyscale value of the mask when you output the images.
#   This is arbitrary, but every material should have its own output color
#   so they can be differentiated
#  
#   **confidence_threshold** - The lower this number, the more voxels will be labled a specific material. Essentially, the ML algorith outptus a confdience value  (centered on 0.5) for every voxel and every material. By default, voxels with  a confidence of 0.5 or greater are determined to be the material in question.  But we can labled voxles with a lower condience level by changing this  parameter
#   
#   **training_image_directory /training_mask_directory**: Input the directory where your training images and masks are located.
# 
#   **validation_fraction**: Input the fraction of images you want to validate your model during training. These are not a independent validation, but are part of the training process.
# 
#   **num_models**: Enter the number of models you want to iteratively train. Because these are statistical models, the performance of any given model will vary. Training more models will allow you to select the model that best fits your data.
#   
#   **num_epochs**: Enter number of epochs that you want to use to train your model. More is generally better, but takes more time.
# 
#   **batch_size**: Input your batch size. Larger batch sizes allow for faster training, but take up more VRAM. If you are running out of VRAM during training, decrease your batch size.
# 
#   **scale**: Input how you want your images scaled during model training and inference. When the scale is 1, your images will be used at full size for training. When the scale is less than 1, your images will be downsized according to the scale you set for training and inference, decreasing VRAM usage. If you run out of VRAM during training, consider rescaling your images.
# 
#   **models_directory**: Directory where your models are saved.
# 
#   **model_group**: Name for the group models you iteratively generate.
# 
#   **current_model_name**: Name for each individual model you generate; will automatically be labeled 1 through n for the number of models you specify above.
# 
#   **val_images/val_masks**: Input the directory where your independent validation images and masks are located. These images are not used for training and are used as an independent validation of your model.
# 
#   **csv_directory**: Directory where a CSV file of your validation results will be saved.
# 
#   **inference_directory**: Directory where the images you want analyzed are located.
# 
#   **output_directory**: Directory where you want your analysis results to be saved.
# 
# 
    

#Creating a list of materials so we can iterate through it
materials = [
             Material("notberry", [0,0,0], 1, 0.5),
             Material("berry", [255,255,255], 100, 0.5),
             ]

# Boolean whether to retrain models or use current models
new_training = False

# Project directory
# IMPORTANT - ALL DIRECTORIES NEED TO END IN A /
proj_dir = "/path/to/project/directory/"
 
num_materials =len(materials)

# #Various input/output directories
# # IMPORTANT: END EACH DIRECTORY PATH WITH A "/"
training_image_directory = proj_dir + "train/images/"
training_mask_directory = proj_dir + "train/masks/"
#
# #Fraction of total annotations you want to leave for validating the model.
# validation_fraction=0.2
# 
# #Model Performance varies, make multiple models to have the best chance at success.
# num_models=7
# 
# #Model Performance improves with increasing epochs, to a point.
# num_epochs=70
# 
# """Increasing batch size increase model training speed, but also eats up VRAM on the GPU. Find a balance between scale and batch size
# that best suits your needs"""
batch_size=3

#Decrease scale to decrease VRAM usage; if you run out of VRAM during traing, restart your runtime and down scale your images
scale=0.8

#Input model directory
# IMPORTANT: END EACH DIRECTORY PATH WITH A "/"
models_directory = proj_dir + "best_models/"

#Input the name you want to use for your group of models
model_group='model_group_name/'

# Input the full model basename, omitting the .pth extension
current_model_name = "full_model_name"


#Input the directory of the data you want to segment here.
inference_directory= proj_dir + 'inferenceImages/'

#Input the 5 alpha-numeric characters proceding the file number of your images
  #EX. Jmic3111_S0_GRID image_0.tif ----->mage_
proceeding="mage_"
#Input the 4 or mor alpha-numeric characters following the file number
  #EX. Jmic3111_S0_GRID image_0.tif ----->.tif
following=".jpg"

output_directory = proj_dir + model_group + 'watershed_adj/'


## Write variables to a file ##

# # List all objects
# objects = locals()
# 
# ## Write all of the above parameters to a python script that will be imported at later stages
# param_filename = proj_dir + "fcn_workflow_parameters_" + ds + ".py"
#
#
# # Open a file
# handle = open(param_filename, "w")
#
# # Write all the parameters to this file
# for key in objects:
#  value = objects[key]
#  # If the object value is one of the following, print it
#  if (type(value).__name__ in ["str", "int", "list"]):
#    if type(value).__name__ == "int":
#      handle.write(key + ' = ' + str(value) + '\n')
#    else:
#      handle.write(key + ' = "' + str(value) + '"\n')
#
# # Close the file
# handle.close()




################################################################################
################################################################################
### DO NOT EDIT BELOW THIS LINE ################################################
################################################################################
################################################################################













# Load packages

from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import random
import albumentations as A
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from scipy.ndimage import morphology
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import matplotlib
import torchvision
from torchvision.models.segmentation.fcn import FCNHead
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import os
import pandas as pd
import sys
import math
import re
import time
import cv2
import argparse
import scipy as scipy
import io as io
from scipy import ndimage as ndi
from skimage import (color, feature, filters, measure, morphology, segmentation, util)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
sys.path.append(os.path.join(sys.path[0]))  # To find local version of the library
from skimage.filters import sobel
from scipy import ndimage
from torchvision.models.detection.rpn import AnchorGenerator
from skimage.color import rgb2grey, label2rgb
from skimage.segmentation import slic, join_segmentations, watershed
from skimage.morphology import binary_erosion, disk
from skimage import io,exposure, feature, filters, io, measure, morphology, restoration, segmentation, transform, util, data, color
from skimage.measure import label, regionprops
from skimage.transform import rescale, resize, downscale_local_mean




# #**Parameter Loading**
 
class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=scale, transform=False):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.transform=transform
        self.t_list=A.Compose([A.HorizontalFlip(p=0.4),A.VerticalFlip(p=0.4), A.Rotate(limit=(-50, 50), p=0.4),])
        self.means=[0]
        self.stds=[1]

        
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
 
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')
 
    def __len__(self):
        return len(self.ids)
 
 
    @classmethod
    def mask_preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))
 
        img_nd = np.array(pil_img)
 
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
 
       
        return img_nd
    
 
        
    def img_preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))
 
        img_nd = np.array(pil_img)
 
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
 
       
 
        return img_nd
 
    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '*')
        img_file = glob(self.imgs_dir + idx + '*')
 
        assert len(mask_file) == 1,             f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1,             f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])
 
  
        
 
        
        #Reshapes from 1 channel to 3 channels in grayscale
        img = self.img_preprocess(img, self.scale)
        mask = self.mask_preprocess(mask, self.scale)
        new_image=np.zeros((img.shape[0],img.shape[1],3))
        new_image[:,:,0]=img[:,:,0]
        new_image[:,:,1]=img[:,:,0]
        new_image[:,:,2]=img[:,:,0]
        
 
 
 
        img=new_img
 
        new_mask = np.zeros((num_materials,img.shape[0],img.shape[1]))
        # print(mask.shape)       
        for i, mat in enumerate(materials):
          # plt.imshow(mask[:,:,0])
          # plt.show()
          indices = np.all(mask == mat.input_rgb_vals, axis=-1)
          new_mask[i,:,:][indices] = 1
 
        mask = new_mask
  
        # plt.imshow(mask[1,:,:])
        # i=6
        # for i in range(len(mask)):
        #   plt.imshow(mask[i,:,:])
        #   plt.show()
        
        if img.max() > 1:
            img = img / 255
 
       
 
        
        if self.transform:
            augmented=self.t_list(image=img, masks=mask)
            img=augmented["image"]
            mask=augmented["masks"]
            
 
        
 
        img = img.transpose((2, 0, 1))
        
        mask=np.array(mask)
        
        
 
        
 
        img=torch.from_numpy(img)
        mask=torch.from_numpy(mask)
        
        img=transforms.Normalize(mean=self.means, std=self.stds)(img)
        return img, mask
        
        
dataset = BasicDataset(training_image_directory, training_mask_directory, scale=scale, transform=False)
 
#!!!!!!!!!!!!!!!!!!!!!!!!!!Set batch size here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# train, val=trainval_split(dataset, val_fraction=0.5)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)#, collate_fn=pad_collate)
#val_loader = DataLoader(val, batch_size=3, shuffle=False, num_workers=0, pin_memory=True)#, collate_fn=pad_collate)
nimages = 0
mean = 0.
std = 0.
for batch, _ in train_loader:
    # Rearrange batch to be the shape of [B, C, W * H]
    batch = batch.view(batch.size(0), batch.size(1), -1)
    # Update total number of images
    nimages += batch.size(0)
    # Compute mean and std here
    mean += batch.mean(2).sum(0) 
    std += batch.std(2).sum(0)
 
# Final step
mean /= nimages
std /= nimages
 
print(mean)
print(std)

dataset.means=mean
dataset.stds=std 


## INFERENCE ##
        

"""Input model number here"""
model_number='5'

# Get the untrained model
model=torchvision.models.segmentation.fcn_resnet101(pretrained=False)
# model.backbone.conv1=nn.Sequential(nn.Conv2d(1,3, (1,1), (1,1), (0,0), bias=False), model.backbone.conv1)

model.classifier=FCNHead(2048, num_materials)
 
# Assign the device
device = torch.device('cuda')


outputs=[]
model.to(device)
 
#!!!!!!!!!!!!!!!!!!!!!Select Correct Model from the best models directory!!!!!!!!!!!!!!!!!!!!!!!!!1
model.load_state_dict(torch.load(models_directory + model_group + current_model_name + model_number + '.pth'), strict=False)

# Initialize the model trainer
model.train()


## IMAGE SEGMENTATION ##

#!!!!!!!!!!!!!!!!!!!!!Put the name of the folder with the images you want to analyze here!!!!!!!!!!!!!!!!!!!!!!!
dir_name = inference_directory

# List the file to make inferences
filenames = os.listdir(dir_name)

print(str(len(filenames)) + " images found.")


file_name=[]
color=[]
value_counts=[]
sort_idx = np.argsort([(int(filename.split(proceeding)[1].split(following)[0])) for filename in filenames])
# sort_idx = np.argsort([(int(filename.split(following)[0])) for filename in filenames])

for i in sort_idx:
    #makes new directory called "(directory name here) + name in red" that your new images go into
    new_dir_name = output_directory
    if not os.path.exists(new_dir_name):
      os.makedirs(new_dir_name)
    
    for mat in materials:
      new_dir_name_mat= new_dir_name + mat.name
      if not os.path.exists(new_dir_name_mat):
        os.makedirs(new_dir_name_mat)
    filename = filenames[i]
    
    image = Image.open(dir_name +'/'+ filenames[i])
    image1=(image)
    image1=np.array(image1)
    image1=rgb2grey(image1)
    image1 = rescale(image1, scale, anti_aliasing=True)
    
    w, h = image.size
    # print(image.size)
    #!!!!!!!!!!!!!!!!!!!!Make sure scale matches!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    scale=scale
    newW, newH = int(scale * w), int(scale * h)
    image=image.resize((newW, newH))
    image1 = image1[0:newH, 0:newW] # Add this to resize image1
    image=np.array(image, dtype=float)
    new_im=np.zeros((3, newH, newW))
    new_im[0,:,:]=image[:,:,0]
    new_im[1,:,:]=image[:,:,1]
    new_im[2,:,:]=image[:,:,2]
    image=new_im
    

    image=torch.from_numpy(image)

#!!!!!!!!!!!!!!!!!!!!!!!!!!Make sure normalization goes match above!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    image=transforms.Normalize(mean=mean, std=std)(image)
 
    image.unsqueeze_(0)
    image = image.to(device=device, dtype=torch.float32)


    tic=time.time()
    with torch.no_grad():
      mask=model(image)['out']
      mask=nn.Sigmoid()(mask)
      # mask=mask.cpu().detach().numpy()
    toc=time.time()
    print('time: '+str(toc-tic))
#!!!!!!!!!!!!!!!!!Make sure there are the same number of mask outputs as you trained on!!!!!!!!!!!!!!!!!!!!!

    image_rescaled=zeros2
    combined_image = image_rescaled 

    list_of_mat_tables = []
    for i, mat in enumerate(materials):
      mat_mask = mask.cpu().detach().numpy()[0,i,:,:]
      
      mat_mask[mat_mask >= mat.confidence_threshold] = mat.output_val
      mat_mask[mat_mask < mat.confidence_threshold] = 0

      mat_mask=np.array(mat_mask, dtype='ubyte')
      combined_image = np.add(combined_image, mat_mask[:,:])

      io.imsave(new_dir_name+'/' + mat.name + '/'+filename.split(following)[0]+'_' + mat.name + "_mask.png", mat_mask)
      np_mat = np.array(mat_mask)
      

      
      label_mat = label(np_mat, background = 0)
      if np.sum(label_mat)<1:
        label_mat=np.ones_like(label_mat)
      elif np.sum(label_mat)>=1:
        label_mat=label_mat

      
      mat_table=measure.regionprops_table(label_mat, image1, properties=['label','area','perimeter'])
      mat_table=pd.DataFrame(mat_table)
      if mat_table['area'].sum()==np.ones_like(label_mat).sum():
        mat_table_a=mat_table['area'].sum()==0
      
      elif mat_table['area'].sum()<np.ones_like(label_mat).sum():
        mat_table_a=mat_table['area'].sum()/(scale**2)

      if mat_table['area'].sum()==np.ones_like(label_mat).sum():
        mat_table_p=mat_table['perimeter'].sum()==0
     
      elif mat_table['area'].sum()<np.ones_like(label_mat).sum():
        mat_table_p=mat_table['perimeter'].sum()/(scale)



      # mat_table_a=mat_table['area'].sum()
      # mat_table_p=mat_table['perimeter'].sum()

      list_of_mat_tables.append(mat_table_a)
      list_of_mat_tables.append(mat_table_p)
      

    io.imsave(new_dir_name+'/'+filename.split(following)[0]+'.png', combined_image)    

    print(list_of_mat_tables)
    list_of_mat_tables = np.array(list_of_mat_tables, dtype=int)
    whole_leaf = value_counts.append(list_of_mat_tables)
    print(whole_leaf)

    name = file_name.append([filename])
  
counts=(np.array(value_counts))
print(counts)
names=(np.array(file_name))
print(names)

whole_leaf=np.concatenate((names, counts), axis=1)

whole_leaf_table=(pd.DataFrame(whole_leaf))
print(whole_leaf_table)
table_columns = [[f"{mat.name} area(pix)", f"{mat.name} perimeter (pix)"] for mat in materials]
print(table_columns)
table_columns = [element for sublist in table_columns for element in sublist] #flattening the list
print(table_columns)
table_columns = ["file_name"] + table_columns
print(table_columns)
whole_leaf_table.columns = table_columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_row',None)
# print(d)
#!!!!!!!!!!!!!!!!!!!!!!!!!!Can change table output name here!!!!!!!!!!!!!!!!!!!!!!!!!!
whole_leaf_table.to_csv(new_dir_name+'/'+'material area and perimeter.csv')
#print (counts)
print('----end-----')
