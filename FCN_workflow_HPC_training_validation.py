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
 
print(newmean)
print(newstd)


# Now you are ready to train your model in code block 4. 
# If you run out of VRAM, restart the runtime, reload google drive, and try again. Also consider rescaling your images or decreasing your batch size.
# 

# #**Model** **Training**
# Please do not alter this code.

# In[ ]:


## Proceed with model training only if the indicator above is True
if new_training:

  #For loop for FCN model training Cell Code Box 4
  #!cd "drive/My Drive/Colab Notebooks"
  # Semantic Segmentation and Data Extraction in Pytorch Using FCN by Pranav Raja and a tiny bit by Devin Rippner (Plant AI and BioPhysics Lab)
  # a work in progress, works well overall but need mroe people to look at it and identify bugs
  #%%
   
  import torchvision
  from torchvision.models.segmentation.fcn import FCNHead
  from torchvision.models.segmentation.deeplabv3 import DeepLabHead
  from torch.utils.data import DataLoader, random_split
  import torch
  # from torch._six import container_abcs, string_classes, int_classes
  import torchvision.transforms as T
  import matplotlib.pyplot as plt
  import torch.nn as nn
  import os
  import psutil
  import gc
  import random
   
  dir_checkpoint = models_directory
   
   
  model_group=model_group
  num_models=num_models
  # Create model group and name paths
  model_group_dir1 = os.path.join(dir_checkpoint, model_group)
  if not os.path.exists(model_group_dir1):
    os.mkdir(model_group_dir1)
  
  seed=0
  torch.manual_seed(seed)
  random.seed(seed)
  np.random.seed(seed)
  for i in range(num_models):
    #!!!!!!! Here we pull in a pretrained FCN on torch and we replace the output layer since we have six classes rather than 21!!!!!!!!
    num_classes=num_materials
    model=torchvision.models.segmentation.fcn_resnet101(pretrained=True, progress=True)
    # model.backbone.conv1=nn.Sequential(nn.Conv2d(1,3, (1,1), (1,1), (0,0), bias=False), model.backbone.conv1)
    model.classifier=FCNHead(2048, num_classes)
    
    def trainval_split(dataset, val_fraction=0.5):
   
      validation_size = int(len(dataset) * val_fraction)
      train_size = len(dataset) - validation_size
      # print(validation_size)
      # print(train_size)
      # print(len(dataset))
      # print(dataset.dataset_size)
      train, val = torch.utils.data.random_split(dataset, [train_size, validation_size], generator=torch.Generator().manual_seed(i))
   
      return train, val
   
   
   
    dataset= BasicDataset(training_image_directory, training_mask_directory, scale=scale, transform=True)
    dataset_train, dataset_val=trainval_split(dataset, val_fraction=validation_fraction)
    #!!!!!select folders for the images and masks associated with training and validation here. Also specify image scaling factor here!!!!!!!!!!!!!!!!
    # dataset_train = BasicDataset(training_image_directory, training_mask_directory, 1, transform=True)
    # dataset_val = BasicDataset(training_image_directory, training_mask_directory, 1, transform=False)
    # dataset_train = BasicDataset("drive/My Drive/FCN WORKFLOW PAPER/train/image_/", "drive/My Drive/FCN WORKFLOW PAPER/train/mask_edited2/", 1, transform=True)
    # dataset_val = BasicDataset("drive/My Drive/FCN WORKFLOW PAPER/test/image_/", "drive/My Drive/FCN WORKFLOW PAPER/test/mask_edited2/", 1, transform=False)
    
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Specify Batch Size Here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)#, collate_fn=pad_collate)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)#, collate_fn=pad_collate)
   
   
    #%%
   
    # this is the train code 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!! Input epochs here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    num_epochs=num_epochs
    # read up on optimizers but Adam should work for now, if you get good results with Adam then you can try SGD (it's harder to tune but usually converges better)
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3)
   
    #just initializing a value called best_loss
    best_loss=999
   
    # choose a loss function
    # criterion=nn.CrossEntropyLoss()
    #criterion=nn.BCELoss().cuda()
    criterion = nn.BCEWithLogitsLoss()
    # class diceloss(nn.Module):
    #     def __init__(self, epsilon):
    #         # super(diceloss, self).init()
    #         super(diceloss, self).__init__()
    #         self.sigmoid=nn.Sigmoid()
    #         self.epsilon=epsilon
    #         # print('HI')
    #     def forward(self, pred, target):
    #         if target.size() != pred.size():
    #             raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), pred.size()))
    #         pred=self.sigmoid(pred)
    #         tp=torch.sum(target*pred, (1,2,3))
    #         fp=torch.sum((1-target)*pred, (1,2,3))
    #         fn=torch.sum(target*(1-pred), (1,2,3))
    #         # precision=tp/(tp+fp)
    #         # recall=tp/(tp+fn)
    #         f1=(tp)/(tp+self.epsilon+0.5*(fp+fn))
    #         # print(f1)
    #         return 1-torch.mean(f1)
    # criterion=diceloss(epsilon=epsilon)
    #
    # model.train()
    # model.train()
    # 
    #this is the train loop
    for epoch in range(num_epochs):
        print(psutil.virtual_memory().percent)
        print('Epoch: ', str(epoch))
      #add back if doing fractional training
        train_loader.dataset.dataset.transform=True
        model.train()
        for images, masks in train_loader:
   
            images = images.to(device=device, dtype=torch.float32)
            masks = masks.to(device=device, dtype=torch.float32)
   
            #forward pass
            preds=model(images)['out'].cuda()
          
            #compute loss
            loss=criterion(preds, masks)
          
            #reset the optimizer gradients to 0
            optimizer.zero_grad()
   
            #backward pass (compute gradients)
            loss.backward()
   
            #use the computed gradients to update model weights
            optimizer.step()
   
            print('Train loss: '+str(loss.to('cpu').detach()))
        # model.eval()
        #add back if doing fractional training
        val_loader.dataset.dataset.transform=False
        current_loss=0
        
        #test on val set and save the best checkpoint
        model.eval()
        with torch.no_grad():
          for images, masks in val_loader:
              images = images.to(device=device, dtype=torch.float32)
              masks = masks.to(device=device, dtype=torch.float32)
              preds=model(images)['out'].cuda()
              # print(preds)
              # print(masks)
              loss=criterion(preds, masks)
              #print('hi')
              current_loss+=loss.to('cpu').detach()
              del images, masks, preds, loss
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!Re-name model here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!        
        if best_loss>current_loss:
            best_loss=current_loss
            print('Best Model Saved!, loss: '+ str(best_loss))
            torch.save(model.state_dict(), dir_checkpoint+model_group + current_model_name+str(i+1)+".pth")
        else:
            print('Model is bad!, Current loss: '+ str(current_loss) + ' Best loss: '+str(best_loss))
        print('\n')
        

# If no retraining, adjust the current model name to reflect the most recent models
else:
  import re
  import os
  models_dir = os.path.join(models_directory, model_group)
  # List the trained models
  trained_models = [x for x in os.listdir(models_dir) if ".pth" in x]
  # Cut the #.pth from the end of each file; find the unique names
  unique_model_name = list(set([re.sub("\d+.pth", "", x) for x in trained_models]))
  # Sort
  unique_model_name.sort()
  # Get the last item (this is the most recent)
  current_model_name = unique_model_name[-1]
  
  


"""# **Validation**
Please do not alter this code
"""

# Recalculate the number of models
import os
models_dir = os.path.join(models_directory, model_group)
num_models = len([x for x in os.listdir(models_dir) if current_model_name in x and ".pth" in x])

import torch.nn as nn
import torchvision
import torch
from torchvision.models.segmentation.fcn import FCNHead
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torch.utils.data import DataLoader, random_split
#from statistics import mean
import numpy as np
import pandas as pd
# interim_list=[]
modeldata = pd.DataFrame(columns=["name", "precision", "recall", "accuracy", "f1"])
# for mat in enumerate(materials):
#   modeldata=pd.DataFrame(columns=['model_group',mat.name + " precision",mat.name + " recall",mat.name + " accuracy",mat.name + " f1"])
# num_models=num_models
 
for s in range(num_models):
  # model=torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)
 
 
  model=torchvision.models.segmentation.fcn_resnet101(pretrained=False)
  # model.backbone.conv1=nn.Sequential(nn.Conv2d(1,3, (1,1), (1,1), (0,0), bias=False), model.backbone.conv1)
  #!!!!!!!!!!!!!!!!!Specify Layer # here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  model.classifier=FCNHead(2048, num_materials)
  # model.classifier=DeepLabHead(2048, 6)
  device = torch.device('cuda')
 
  outputs=[]
  model.to(device)
 
  #!!!!!!!!!!!!!!!!!!!!!Select Correct Model from the best models directory!!!!!!!!!!!!!!!!!!!!!!!!!1
 
  model.load_state_dict(torch.load(models_directory+model_group + current_model_name+str(s+1)+".pth"), strict=False)
 
 
  model.train()
 
  dataset_val = BasicDataset(test_images, test_masks, scale=scale, transform=False)
  val_loader = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)#, collate_fn=pad_collate)
 
  prop_list = []
  for mat in materials:
    prop_list.append([[],[],[],[]])
 
  for images, target in val_loader:
    images = images.to(device=device, dtype=torch.float32)
    target = target.to(device=device, dtype=torch.float32)
 
    with torch.no_grad():
      pred=model(images)['out'].cuda()
      pred=nn.Sigmoid()(pred)
    
    for i, mat in enumerate(materials):
      material_target=target[:,i,:,:]
      material_pred = pred[:, i, :, :]
      material_pred[material_pred >=mat.confidence_threshold] = 1
      material_pred[material_pred <=mat.confidence_threshold] = 0
      pred[:, i, :, :]=material_pred
 
      material_tp=torch.sum(material_target*material_pred, (1,2))
      material_fp=torch.sum((1-material_target)*material_pred, (1,2))
      material_fn=torch.sum(material_target*(1-material_pred), (1,2))
      material_tn=torch.sum((1-material_target)*(1-material_pred), (1,2))
 
     
 
      material_precision=torch.mean((material_tp+0.000000001)/(material_tp+material_fp+0.000000001))
      material_recall=torch.mean((material_tp+0.000000001)/(material_tp+material_fn+0.000000001))
      material_accuracy=torch.mean((material_tp+material_tn+0.000000001)/(material_tp+material_tn+material_fp+material_fn+0.000000001))
      material_f1=torch.mean(((material_tp+0.000000001))/(material_tp++0.000000001+0.5*(material_fp+material_fn)))
 
    
      prop_list[i][0].append(material_precision.cpu().detach().numpy())
      prop_list[i][1].append(material_recall.cpu().detach().numpy())
      prop_list[i][2].append(material_accuracy.cpu().detach().numpy())
      prop_list[i][3].append(material_f1.cpu().detach().numpy())
 
          
 
 
 
  # print(current_model_name+str(s+1))
  model_name=current_model_name
  model_number=(str(s+1))
  print(model_name)
 
  #printing with pandas
  properties = {"name" : [mat.name for mat in materials],
                "precision" : [str(np.mean(prop_list[i][0])) for i in range(num_materials)],
                "recall" : [str(np.mean(prop_list[i][1])) for i in range(num_materials)],
                "accuracy" : [str(np.mean(prop_list[i][2])) for i in range(num_materials)],
                "f1" : [str(np.mean(prop_list[i][3])) for i in range(num_materials)]}
  df = pd.DataFrame(properties, columns = ["name", "precision", "recall", "accuracy", "f1"])
  df=pd.concat([df, pd.DataFrame(columns=["model number","model name"])])
  df[["model number","model name"]]=[model_number, model_name]
  # display(df)
  
  modeldata=modeldata.append([df], ignore_index=True)


 
#   for i, mat in enumerate(materials):
#     precision_final = np.mean(prop_list[i][0])
#     print(mat.name + " precision: " + str(precision_final))
#     recall_final = np.mean(prop_list[i][1])
#     print(mat.name + " recall: " + str(recall_final))
#     accuracy_final = np.mean(prop_list[i][2])
#     print(mat.name + " accuracy: " + str(accuracy_final))
#     f1_final = np.mean(prop_list[i][3])
#     print(mat.name + " f1: " + str(f1_final))
#     # modeldata1=modeldata.append({'name': mat.name, mat.name + " precision": precision_final, mat.name + " recall": recall_final, mat.name + " accuracy": accuracy_final, mat.name + " f1": f1_final}, ignore_index=True)
#     # model_data=modeldata.append(modeldata1)
# # model_data=df.append(interim_list)
# print(modeldata)
# md=pd.concat([modeldata, pd.DataFrame(columns=["model name"])])
# md[["model name"]]=[current_model_name]
# pd.set_option("display.max_rows", None, "display.max_columns", None)
# display(modeldata)

#   display(modeldata)

"""#**Save Validation CSV**
Please do not alter this code
"""
# display(modeldata)
modeldata.to_csv(csv_directory)
