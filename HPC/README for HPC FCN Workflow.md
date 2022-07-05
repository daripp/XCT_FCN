FCNWorkflow - SCINet Implementation
The code in this repository works on USDA's SCINet Scientific Computing resource.

Setup
To make sure the correct version of python packages are used on SCINet, it is advised that a virtual environment be created. Instructions from SCINet on creating python virtual environments can be found here.

For the sake of ease, two scripts are provided in this repository that create virtual environments depending on the CUDA computing platform that is available. As of today (April 13, 2022), when running jobs on the scavenger-gpu SCINet partition, the CUDA platform is version 11.3.

To set up the virtual environment for CUDA version 11.3 on SCINet, follow these steps:

Start an interactive session on SCINet. It does not need to be on a GPU partition, but it should not be on the login node.
Navigate to the directory where you want to establish the virtual environment. It may be the same as your project working directory.
Run the following script for setting up the virtual environment:
sh setup_virtenv_cuda113.sh
The name of the virtual environment that is created is virtenv_cuda113.

Usage
Two sets of code are provided for i) training and validating the models and ii) deploying models in a production pipeline. Instruction for each are below.

Model Training
To train and validate models, edit the FCN_workflow_SciNet_training_validation.py script. This script contains instructions for the variables that need to be edited by a user.

Then, use the run_FCN_workflow_SciNet_training.sh script to submit a job on SCINet. This .sh script will run the FCN_workflow_SciNet_training_validation.py script.

sbatch run_FCN_workflow_SciNet_training.sh
Model Production
Once you have a model trained, you can use the production version of the scripts for routine prediction. Use the FCN_workflow_SciNet_prediction.py and the run_FCN_workflow_SciNet_prediction.sh scripts in the same way as the model training scripts.

sbatch run_FCN_workflow_SciNet_prediction.sh
