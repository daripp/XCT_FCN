# Welcome to workflow associated with the manuscript: A workflow for segmenting soil and plant X-ray CT images with deep learning in Google’s Colaboratory

X-ray micro-computed tomography (X-ray µCT) has enabled the characterization of the properties and processes that take place in plants and soils at the micron scale. Despite the widespread use of this advanced technique, major limitations in both hardware and software limit the speed and accuracy of image processing and data analysis. Recent advances in machine learning, specifically the application of convolutional neural networks to image analysis have enabled rapid and accurate segmentation of image data. Yet, challenges remain in applying convolutional neural networks to the analysis of environmentally and agriculturally relevant images. Specifically, there is a disconnect between the computer scientists and engineers, who build these AI/ML tools, and the potential end users in agricultural research, who may be unsure of how to apply these tools in their work. Additionally, the computing resources required for training and applying deep learning models are unique, more common to computer gaming systems or graphics design work, rather than traditional computational systems. To navigate these challenges, we developed a modular workflow for applying convolutional neural networks to X-ray µCT images, using low-cost resources in Google’s Colaboratory web application (https://github.com/daripp/XCT_FCN/blob/main/For_publication_Leaf_ALS_FCN_Workflow.ipynb). 

Watch the instructional Youtube video to get started here:
https://youtu.be/JLAZBWDTP1c

# *** Read Me ***



**1. Mount google drive by running  #Code Box 1**

```
#Code Box 1
from google.colab import drive
drive.mount('/content/drive')
!ls "/content/drive/My Drive}
```
**2. Check GPU Allocation:**
GPU's with less VRAM (K80) will need to train on smaller images set at smaller batch sizes than GPU's with more VRAM (A100)

```
#Code Box 2
!!nvidia-smi
```
**3. Specficy material names, input, output, and save directories in code box 3**

  a. Material ***name*** is arbitrary, but is used for naming output folders
  
  b. Input ***[rgb values]*** from training masks; can also accept gray scale values
  
  c. input desired ***gray scale output*** for each material; a little broken when sum of all gray scale values is >255.
  
  d. input desired ***confidence threshold*** for model predictions. The closer to 1, the great the confidence threshold leading to more stringent predictions, the closer to 0,      the lower the confidence threshold, leading to less stringet predictions.
  

```
materials = [
             Material("background", [85,85,85], 30, 0.5),
             Material("epidermis", [170,170,170], 150, 0.5),
             Material("mesophyll", [255,255,255], 255, 0.5),
             Material("air_space", [0,0,0], 1, 0.4),
             Material("bundle_sheath_extension", [103,103,103], 100, 0.5),
             Material('vein', (35,35,35), 180, 0.5)
            ]
```
e. Specify directories for training images and masks:
```
training_image_directory = "drive/MyDrive/ALS Workflow/train/leaf_images/"
training_mask_directory = "drive/MyDrive/ALS Workflow/train/leaf_masks/"
```

f. Specify fraction of training images and masks to be used for validation
```
validation_fraction=0.2
```

g. Specify the number of models you would like to generate
```
num_models=10
```

h. Specify the number of epochs (iterations) you would like use for each model.
```
num_epochs=100
```

i. Specify the batch size. For large images or differently sized images, choose a batch size of 1. For smaller images or images that are the exact same size, you can choose a batch size larger than 1; this will speed up model training but also consume more VRAM.
```
batch_size=1
```

j. Specify the image scale you want to use for training. We recommend keeping the scale at 1, however, large images may require downscaling. This means you will have to decrease the scale to a decimal value between 0 and 1. The more you downscale, the less VRAM you will use, but the more information you will lose for training purposes.

```
scale=1
```
k. Specify the directory where you want to save models
```
models_directory = "drive/MyDrive/ALS Workflow/best_models/"
```
l.Specify the name you want to give the folder that contains the group of models that you will generate based on the number of models you are training
```
model_group='10 leaf bce p2 100 epoch/'
```
m. Specify the name you want to give each individual model with in the model group folder
```
current_model_name = '10 leaf bce p2 100 epoch__'
```
n.Specify the directories where your test images and masks are located
```
test_images = "drive/MyDrive/ALS Workflow/test/test_images/"
test_masks= 'drive/MyDrive/ALS Workflow/test/test_masks/'
```
o. Specify the directory where you want to save a csv file of your model testing results
```
csv_directory = "drive/MyDrive/ALS Workflow/10 leaf bce p2 100 epoch testing code change.csv"
```
p. Specify the directory of the data you want to segment and extract data from after identifying the best model for your data
```
inference_directory= "drive/MyDrive/ALS Workflow/test/test_images/"
```
q. Input the 5 alpha-numeric characters proceding the file number of your images
```
  #EX. Jmic3111_S0_GRID image_0.tif ----->mage_
proceeding="lice_"
```
r. Input the 4 or more alpha-numeric characters following the file number
```
  #EX. Jmic3111_S0_GRID image_0.tif ----->.tif
following=".tif"
```
s. Specify the output directory for your segmentation results
```
output_directory = "drive/MyDrive/ALS Workflow/test_images_results change/"
```

