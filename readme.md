
# XSleepNet: Hydrid End-to-End Sequence-to-Sequence Network for Automatic Sleep Staging

These are source code and experimental setup for XSleepNet based on the __MASS database__.

How to run:
-------------

1. Data preparation with Matlab
- Add directory `./data_preparation/` to Matlab path
- Run `main_data_preparation.m`

2. Network training and testing
- Change directory to a specific network in `./tensorflow/xsleepnet/`
- Run bash scripts,  `bash run_1chan.sh` for 1-channel EEG,  `bash run_2chan.sh` for 2-channel EEG+EOG, and `bash run_3chan.sh` for 3-channel EEG+EOG+EMG. These scripts are demonstrated for running the 1st cross-validation fold. They can be modified to run more folds (20 in total for MASS database setup).
  
_Note:_ You may want to modify and script to make use of your computational resources, such as placing on a specific GPU, you may want to modify the variable `CUDA_VISIBLE_DEVICES="0,-1"`. 

3. Evaluation in Matlab
- Add directory `./evaluation/` to Matlab path
- Run `compute_performance_xsleepnet.m`

Environment:
-------------
- Matlab v7.3 (for data preparation)
- Python3
- Tensorflow GPU 1.13 (for network training and evaluation)
- numpy
- scipy
- sklearn
- h5py

Contact:
-------------
Huy Phan 
Email: huy.phan@ieee.org or h.phan@kent.ac.uk  
