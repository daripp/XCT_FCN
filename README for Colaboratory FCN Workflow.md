# Welcome to the workflow associated with the manuscript: A workflow for segmenting soil and plant X-ray CT images with deep learning in Google’s Colaboratory

X-ray micro-computed tomography (X-ray µCT) has enabled the characterization of the properties and processes that take place in plants and soils at the micron scale. Despite the widespread use of this advanced technique, major limitations in both hardware and software limit the speed and accuracy of image processing and data analysis. Recent advances in machine learning, specifically the application of convolutional neural networks to image analysis have enabled rapid and accurate segmentation of image data. Yet, challenges remain in applying convolutional neural networks to the analysis of environmentally and agriculturally relevant images. Specifically, there is a disconnect between the computer scientists and engineers, who build these AI/ML tools, and the potential end users in agricultural research, who may be unsure of how to apply these tools in their work. Additionally, the computing resources required for training and applying deep learning models are unique, more common to computer gaming systems or graphics design work, rather than traditional computational systems. To navigate these challenges, we developed a modular workflow for applying convolutional neural networks to X-ray µCT images, using low-cost resources in Google’s Colaboratory web application (https://github.com/daripp/XCT_FCN/blob/main/For_publication_Leaf_ALS_FCN_Workflow.ipynb). 

Watch the instructional Youtube video to get started here:https://www.youtube.com/watch?v=_5AIN8Wm-PQ
(depreciated)https://youtu.be/JLAZBWDTP1c

See pre-print for the publication here:
https://doi.org/10.48550/arXiv.2203.09674

Access training and testing data here:
https://data.nal.usda.gov/dataset/x-ray-ct-data-semantic-annotations-paper-workflow-segmenting-soil-and-plant-x-ray-ct-images-deep-learning-google%E2%80%99s-colaboratory

Citation:
Rippner, D.A., Raja, P., Earles, J.M., Buchko, A., Momayyezi, M., Duong, F., Parkinson, D., Forrestel, E., Shackel, K., McElrone, A.J., 2022. A workflow for segmenting soil and plant X-ray CT images with deep learning in Googles Colaboratory. arXiv:2203.09674 [physics].

# *** Read Me ***



# **1. Mount google drive by running  Code Box 1**

```
#Code Box 1
from google.colab import drive
drive.mount('/content/drive')
!ls "/content/drive/My Drive}
```
# **2. Check GPU Allocation:**
GPU's with less VRAM (K80) will need to train on smaller images set at smaller batch sizes than GPU's with more VRAM (A100)

```
#Code Box 2
!!nvidia-smi
```
# **3. Specficy material names, input, output, and save directories in Code Box 3**

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
l. Specify the name you want to give the folder that contains the group of models that you will generate based on the number of models you are training
```
model_group='10 leaf bce p2 100 epoch/'
```
m. Specify the name you want to give each individual model with in the model group folder
```
current_model_name = '10 leaf bce p2 100 epoch__'
```
n. Specify the directories where your test images and masks are located
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
r. Input the 4 or more alpha-numeric characters following the file number. You can use almost any image type, but .tif and .png are best.
```
  #EX. Jmic3111_S0_GRID image_0.tif ----->.tif
following=".tif" 
```
s. Specify the output directory for your segmentation results
```
output_directory = "drive/MyDrive/ALS Workflow/test_images_results change/"
```
# ***Parameter Loading***

Once you have specificied your input/output directories, you can proceed with running this code block (Code box 4). This code block is used for batch normalization purposes; the results of this block are unique to your training data set. When using models for segmentation, you must link to the original training data set you used for model generation or save the 3 tensor means and std values that are generated in the output.

# ***Model Training***

Run Code Box 5 after you have specified a unique model group name. You must run the parameter loading block (Code Box 4) before running model training. If block returns an error such as you have run out of vram, you must restart the notebook and either decrease you image scale or batch size until you do not run out of VRAM.

# ***Validation***

Once your models have trained, it is time to validate them by running Code Box 6. If you interupt your model training to evaluate the model, you must change the number of models to match the number of models you trained. Models are evaluated on their ***precision*** (total correct pixels predictions out of total pixel predictions for a specific material class), ***recall*** (total correct pixel predictions out of all possible correct pixel predictions for a material class), ***accuracy*** (total correct pixel predictions out of all pixel predictions in the image) and ***F1 score*** (the harmonic mean of recall and precesion). You can adjust your predictions by increasing or decreasing your confidence threshold for a particular material class in Code Box 3.

# ***Save Validation***

You can save your validation data by running Code Box 7. It will overwrite the file designated in the ***csv_directory*** input area of Code Box 3.

# Input Best Model Number

Based on your validation results, choose the model that you think does the best job on your validation data and input the model number in Code Box 8.

```
#Code Box 8
"""Input model number here"""
model_number='8'
```

# ***Load Model***

Run Code Box 9 to load the model you selected

# ***Image Segmentation and Data Extraction***

Runc Cod Box 10 to segment your data and extact area and perimeter information from your image data. Binary masks of the prediced materials will be generated as well. These can be operated on independently. A CSV file of the area and perimeter data will automatically be generated after all images in the inference directory in Code Box 3 have been analyzed. If needed, the code can be easily modified to extract other parameters using the region propertiers function.
