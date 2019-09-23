clear all
close all
clc

addpath(genpath('evaluation'));

% Nfold = 20; % MASS database
Nfold = 1; % only on the first fold
yh = cell(Nfold,1);
yt = cell(Nfold,1);
mat_path = './mat/';
% for loading groundtruth labels
listing = dir([mat_path, '*_eeg.mat']);
% load data split
load('./data_split.mat');

for fold = 1 : Nfold
    fold
    % test subjects of this fold
    test_s = test_sub{fold};
    % the number of epochs for every subjects
    sample_size = zeros(numel(test_s), 1); 
    for i = 1 : numel(test_s)
        i
        sname = listing(test_s(i)).name;
        load([mat_path,sname], 'label');
        % this is actual output of the network as we excluded those at the
        % recording ends which do not consitute a full sequence
        sample_size(i) = numel(label) -  (seq_len - 1); 
        % pool ground-truth labels of all test subjects
        yt{fold} = [yt{fold}; double(label)];
    end
    
    % load network output
    load(['./tensorflow/xsleenet/xsleenet_1chan/n',num2str(fold),'/test_ret.mat']);
    
    % as we shifted by one PSG epoch when generating sequences, L (sequence
    % length) decisions are available for each PSG epoch. This segment is
    % to aggregate the decisions to derive the final one.
    score_ = cell(1,seq_len);
    for n = 1 : seq_len
        score_{n} = softmax(squeeze(score(n,:,:)));
    end
    score = score_;
    clear score_;

    for i = 1 : numel(test_s)
        % start and end positions of current test subject's output
        start_pos = sum(sample_size(1:i-1)) + 1;
        end_pos = sum(sample_size(1:i-1)) + sample_size(i);
        score_i = cell(1,seq_len);
        for n = 1 : seq_len
            score_i{n} = score{n}(start_pos:end_pos, :);
            N = size(score_i{n},1);
            % padding ones for those positions not constituting full
            % sequences
            score_i{n} = [ones(seq_len-1,5); score{n}(start_pos:end_pos, :)];
            score_i{n} = circshift(score_i{n}, -(seq_len - n), 1);
        end

        % multiplicative probabilistic smoothing for aggregation
        % which equivalent to summation in log domain
        fused_score = log(score_i{1});
        for n = 2 : seq_len
            fused_score = fused_score + log(score_i{n});
        end
        
        % the final output labels via likelihood maximization
        yhat = zeros(1,size(fused_score,1));
        for k = 1 : size(fused_score,1)
            [~, yhat(k)] = max(fused_score(k,:));
        end

        % pool outputs of all test subjects
        yh{fold} = [yh{fold}; double(yhat')];
    end
end
yh = cell2mat(yh);
yt = cell2mat(yt);

[acc, kappa, f1, ~, spec] = calculate_overall_metrics(yt, yh);
[sens, sel]  = calculate_classwise_sens_sel(yt, yh);
mean_sens = mean(sens);
mean_sel = mean(sel);
