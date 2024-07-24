# Project Name

## Machine learning and Multimodal Deep learning Classifiers for aneurysm rupture risk prediction 

# Project files

The project has two folder, code and results, 

the training and testing code files are in the code folder,

results and trained models are in the result folder,

#### The requirements file contains the libraries used in this project, you need to install these libraries in order to test the code

the files and folder in results folder been explained below

## Results folder

### Datasets folder: 
contians the processed data file, 2000 points training data file is too large, so can please download through onedrive: https://leeds365-my.sharepoint.com/:f:/g/personal/sc20wz_leeds_ac_uk/Epe_HmD9wRpDvcMjsjm5DCsBYVcYJ3aW-nNjxXE2niLkyg?e=GnhxH6

After download, please put the folder in result and code folder in order to run the functions

### final models: 
contains all the trained models in this project

### pynote_results: 
includes the python notebook files that been runned, and result been shown in files

### training_trends: 
includes some of the training loss and accuracy graph file

### pointnetfunct: 
include all the python function that used in this project to build, train and test the model

### DNN.ipynb: 
a python notebook file can be run and show example of 3-layer neural network result

### Machine learning methods.ipynb.ipynb: 
a python notebook file can be run and show example of different Machine learning methods result

### run_pointnet_resul.ipynb: 
can run different trained models and show the results using function format, but need to import the test_application.py first: 

1. import functions ```import test_application as ta ```
2. use functions
```
   ta.output_result(points = 2000, # number of point cloud points
        sample = "uniform", # point cloud sampling methods: uniform/ppd
        model_type = "pointnet", # model name: pointnet/pointnet_2branch/pointnet_3branch
        cuttype = "dome", # cut type: dome/cut1
        model_file = "./final_models/2000points/dome/pointnet_dome_uniform_2000pt.pth") # model file path`)
   ```
## Code folder

The code folder has similiar files with result folder, additionally  with some other files when training the model:

### single_case.ipynb
The python notebook use to show a single cases of model prediction

### Other ipynotebook files

Contains some of the experiment been done in the project, e.g. ablation experiments, internal test experiments, dataset creation
