This script includes the necessary data and scripts for training and evaluating the models using the [Snellius National Supercomputer](https://servicedesk.surf.nl/wiki/display/WIKI/Snellius)


After gaining access to Snellius, this directory can be copied to MobaXTerm and the job-scripts can be called. 
To test and evaluate the models further, adapt the **subset_evaluator.py** and **Training.py** scripts.

Example job scripts can be found at **Recommendation_Training_Evaluation_HPC/job-scripts**
The paths in the scripts need to be adapted and the correct email has to be set
Do some pre-testing to find the appropriate values for partitions, time etc.




**Remark**: Please add the trained models and preprocessed data to the respective parts at:
- Recommendation_Training_Evaluation_HPC/Models/base
- Recommendation_Training_Evaluation_HPC/data
The respective functions to create the data for those modules can be found in **/InitPreprocessing**

The training and loading is shown in the **InitPreprocessing/scripts/main.py** script