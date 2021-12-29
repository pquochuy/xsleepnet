% experiment with recordings cut off from light_on time to light_off time

% NOTE: meta information (e.g. light on time, light off time) will be
% needed to process the edf files by the script preprare_raw_data.m . These
% information is given in the directory __sleepedf78_meta_info. They should
% be placed into the directory where the data is stored.


clear all
close all
clc

addpath(genpath('data_processing'))
addpath(genpath('evaluation'))
addpath(genpath('metrics'))

%% Data preparation%%
% extract eeg, eog, emg channels from edf files 
preprare_raw_data;
% prepare time-frequency image and raw data in epochs
prepare_data;
% generate list of files
gen_file_list;

%% Train and test the networks (run bash scripts to each network's directory) %%

%% Compute performance metrics %%
% this is an example for xsleepnet2 with 1-channel case 
[acc, kappa, f1, sens, spec, classwise_sens, classwise_sel, C] = aggregate_performance('xsleepnet2', 1);