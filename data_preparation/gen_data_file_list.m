
% This script is to generate file containing data list for Tensorflow
% network training

clear all
close all
clc

load('./data_split.mat');

mat_path = './mat/';
Nfold = 20;

% this is a subset (10 subjects) randomly sampled from the training set that will be used in XSleepNet to
% measure the ratio of convergence and overfitting. It is expensive to
% compute this ratio on entire training set, so randomly sampling a small
% subsets would be good enough
train_check_sub = cell(Nfold,1);
for s = 1 : Nfold
    rp = randperm(numel(train_sub{s}));
    train_check_sub{s} = sort(rp(1:10));
end


%% EEG
listing = dir([mat_path, '*_eeg.mat']);
tf_path = '../tf_data/eeg/';
if(~exist(tf_path, 'dir'))
    mkdir(tf_path);
end

for s = 1 : Nfold

    disp(['Fold: ', num2str(s),'/',num2str(Nfold)]);
    
	train_s = train_sub{s};
    eval_s = eval_sub{s};
    test_s = test_sub{s};
    train_check_s = train_sub(train_check_sub{s});
    
    train_filename = [tf_path, 'train_list_n', num2str(s),'.txt'];
    fid = fopen(train_filename,'wt');
    for i = 1 : numel(train_s)
        sname = listing(train_s(i)).name;
        load([mat_path,sname], 'label');
        num_sample = numel(label);
        file_path = ['../../mat/', sname]; % data file is saved in two levels higher than tensorflow code
        fprintf(fid, '%s\t%d\n', file_path, num_sample);
    end
    fclose(fid);
    clear fid file_path
    
    train_check_filename = [tf_path, 'train_list_check_n', num2str(s),'.txt'];
    fid = fopen(train_check_filename,'wt');
    for i = 1 : numel(train_check_s)
        sname = listing(train_check_s(i)).name;
        load([mat_path,sname], 'label');
        num_sample = numel(label);
        file_path = ['../../mat/',sname]; % data file is saved in two levels higher than tensorflow code
        fprintf(fid, '%s\t%d\n', file_path, num_sample);
    end
    fclose(fid);
    clear fid file_path
    
    test_filename = [tf_path, 'test_list_n', num2str(s),'.txt'];
    fid = fopen(test_filename,'wt');
    for i = 1 : numel(test_s)
        sname = listing(test_s(i)).name;
        load([mat_path,sname], 'label');
        num_sample = numel(label);
        file_path = ['../../mat/',sname]; % data file is saved in two levels higher than tensorflow code
        fprintf(fid, '%s\t%d\n', file_path, num_sample);
    end
    fclose(fid);
    clear fid file_path
    
    eval_filename = [tf_path, 'eval_list_n', num2str(s),'.txt'];
    fid = fopen(eval_filename,'wt');
    for i = 1 : numel(eval_s)
        sname = listing(eval_s(i)).name;
        load([mat_path,sname], 'label');
        num_sample = numel(label);
        file_path = ['../../mat/',sname]; % data file is saved in two levels higher than tensorflow code
        fprintf(fid, '%s\t%d\n', file_path, num_sample);
    end
    fclose(fid);
    clear fid file_path
end


%% EOG
listing = dir([mat_path, '*_eog.mat']);
tf_path = '../tf_data/eog/';
if(~exist(tf_path, 'dir'))
    mkdir(tf_path);
end

for s = 1 : Nfold

    disp(['Fold: ', num2str(s),'/',num2str(Nfold)]);
    
	train_s = train_sub{s};
    eval_s = eval_sub{s};
    test_s = test_sub{s};
    train_check_s = train_sub(train_check_sub{s});
    
    train_filename = [tf_path, 'train_list_n', num2str(s),'.txt'];
    fid = fopen(train_filename,'wt');
    for i = 1 : numel(train_s)
        sname = listing(train_s(i)).name;
        load([mat_path,sname], 'label');
        num_sample = numel(label);
        file_path = ['../../mat/',sname];
        fprintf(fid, '%s\t%d\n', file_path, num_sample);
    end
    fclose(fid);
    clear fid file_path
    
    train_check_filename = [tf_path, 'train_list_check_n', num2str(s),'.txt'];
    fid = fopen(train_check_filename,'wt');
    for i = 1 : numel(train_check_s)
        sname = listing(train_check_s(i)).name;
        load([mat_path,sname], 'label');
        num_sample = numel(label);
        file_path = ['../../mat/',sname]; 
        fprintf(fid, '%s\t%d\n', file_path, num_sample);
    end
    fclose(fid);
    clear fid file_path
    
    test_filename = [tf_path, 'test_list_n', num2str(s),'.txt'];
    fid = fopen(test_filename,'wt');
    for i = 1 : numel(test_s)
        sname = listing(test_s(i)).name;
        load([mat_path,sname], 'label');
        num_sample = numel(label);
        file_path = ['../../mat/',sname];
        fprintf(fid, '%s\t%d\n', file_path, num_sample);
    end
    fclose(fid);
    clear fid file_path
    
    eval_filename = [tf_path, 'eval_list_n', num2str(s),'.txt'];
    fid = fopen(eval_filename,'wt');
    for i = 1 : numel(eval_s)
        sname = listing(eval_s(i)).name;
        load([mat_path,sname], 'label');
        num_sample = numel(label);
        file_path = ['../../mat/',sname];
        fprintf(fid, '%s\t%d\n', file_path, num_sample);
    end
    fclose(fid);
    clear fid file_path
end


%% EMG
listing = dir([mat_path, '*_emg.mat']);
tf_path = '../tf_data/emg/';
if(~exist(tf_path, 'dir'))
    mkdir(tf_path);
end

for s = 1 : Nfold

    disp(['Fold: ', num2str(s),'/',num2str(Nfold)]);
    
	train_s = train_sub{s};
    eval_s = eval_sub{s};
    test_s = test_sub{s};
    train_check_s = train_sub(train_check_sub{s});
    
    train_filename = [tf_path, 'train_list_n', num2str(s),'.txt'];
    fid = fopen(train_filename,'wt');
    for i = 1 : numel(train_s)
        sname = listing(train_s(i)).name;
        load([mat_path,sname], 'label');
        num_sample = numel(label);
        file_path = ['../../mat/',sname];
        fprintf(fid, '%s\t%d\n', file_path, num_sample);
    end
    fclose(fid);
    clear fid file_path
    
    train_check_filename = [tf_path, 'train_list_check_n', num2str(s),'.txt'];
    fid = fopen(train_check_filename,'wt');
    for i = 1 : numel(train_check_s)
        sname = listing(train_check_s(i)).name;
        load([mat_path,sname], 'label');
        num_sample = numel(label);
        file_path = ['../../mat/',sname]; 
        fprintf(fid, '%s\t%d\n', file_path, num_sample);
    end
    fclose(fid);
    clear fid file_path
    
    test_filename = [tf_path, 'test_list_n', num2str(s),'.txt'];
    fid = fopen(test_filename,'wt');
    for i = 1 : numel(test_s)
        sname = listing(test_s(i)).name;
        load([mat_path,sname], 'label');
        num_sample = numel(label);
        file_path = ['../data_processing/mat/',sname];
        fprintf(fid, '%s\t%d\n', file_path, num_sample);
    end
    fclose(fid);
    clear fid file_path
    
    eval_filename = [tf_path, 'eval_list_n', num2str(s),'.txt'];
    fid = fopen(eval_filename,'wt');
    for i = 1 : numel(eval_s)
        sname = listing(eval_s(i)).name;
        load([mat_path,sname], 'label');
        num_sample = numel(label);
        file_path = ['../data_processing/mat/',sname];
        fprintf(fid, '%s\t%d\n', file_path, num_sample);
    end
    fclose(fid);
    clear fid file_path
end



