# Experiment Execution Guide

Here I shortly explain how to run experiments using the python binding to mujoco-mpc.
## IMPORTANT

After each change in the C++ code, you have to do a new build and also run 
```bash
python setup.py install
```
This has to be done after each change again !!!

## Run Experiment Script

This file runs the actual experiment, the results are stored in a folder. 
Keep in mind, all the parameters (planner, horizon, cost weights, etc.) are loaded from the xml file. Someone might have changed them in the last pull. And they might be different for different tasks. 

## Evaluation Script
This script generates the plots. It will need the paths to the folder where the results are stored. They are printed at the end of the experiment script.

## Compare Results
In this folder is the json file with the experiment data from the humanoid_bench paper. I added a few lines of python to load the data and plot it. 