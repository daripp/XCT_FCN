# Welcome to workflow associated with the manuscript: A workflow for segmenting soil and plant X-ray CT images with deep learning in Google’s Colaboratory

X-ray micro-computed tomography (X-ray µCT) has enabled the characterization of the properties and processes that take place in plants and soils at the micron scale. Despite the widespread use of this advanced technique, major limitations in both hardware and software limit the speed and accuracy of image processing and data analysis. Recent advances in machine learning, specifically the application of convolutional neural networks to image analysis have enabled rapid and accurate segmentation of image data. Yet, challenges remain in applying convolutional neural networks to the analysis of environmentally and agriculturally relevant images. Specifically, there is a disconnect between the computer scientists and engineers, who build these AI/ML tools, and the potential end users in agricultural research, who may be unsure of how to apply these tools in their work. Additionally, the computing resources required for training and applying deep learning models are unique, more common to computer gaming systems or graphics design work, rather than traditional computational systems. To navigate these challenges, we developed a modular workflow for applying convolutional neural networks to X-ray µCT images, using low-cost resources in Google’s Colaboratory web application (https://github.com/daripp/XCT_FCN/blob/main/For_publication_Leaf_ALS_FCN_Workflow.ipynb). 

Watch the instructional Youtube video to get started here:
https://youtu.be/JLAZBWDTP1c

#Instructions:
Mount google drive by running  #Code Box 1

```json
   #Code Box 1
from google.colab import drive
drive.mount('/content/drive')
!ls "/content/drive/My Drive
```
