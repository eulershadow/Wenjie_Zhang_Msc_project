### Project files
The project has been slipt into two folder code and result

In result folder:

Datasets folder: contians the processed data file, 2000 points training data file is too largre, so can be download through onedrive: https://leeds365-my.sharepoint.com/:f:/g/personal/sc20wz_leeds_ac_uk/Epe_HmD9wRpDvcMjsjm5DCsBYVcYJ3aW-nNjxXE2niLkyg?e=GnhxH6

final models: contains all the trained models in this project

pynote_results: includes the python notebook files that been runned, and result been shown in files

training_trends: includes some of the training loss and accuracy graph file

pointnetfunct: include all the python function that used in this project to build, train and test the model

DNN.ipynb: a python notebook file can be run to show example of 3-layer neural network result

run_pointnet_result: can run different trained models and show the results using function format, but need to import the test_application.py first: import test_application as ta ta.output_result(points = 2000, # number of point cloud points sample = "uniform", # point cloud sampling methods: uniform/ppd model_type = "pointnet", # model name: pointnet/pointnet_2branch/pointnet_3branch cuttype = "dome", # cut type: dome/cut1 model_file = "./final_models/2000points/dome/pointnet_dome_uniform_2000pt.pth") # model file path
