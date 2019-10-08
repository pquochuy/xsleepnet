clear all
close all
clc

% path to raw PSG data which contains raw .mat data files each for one subject
%% 
% data should be saved in 'data' variable. With the sampling freq of 100Hz, 
% 'data' should have size Nx3000xnum_chan
% - N: the number of epochs
% - 3000: the number of samples in 30s epoch at the sampling rate of 100
% - num_chan: the number of channels (here 1st, 2nd, and 3rd channels
% correspond to EEG, EOG, and EMG, respectively)
% Sleep stages corresponding to the PSG epochs should be stored in 'labels'
% variable of size Nx1 and the values in set {1,2,3,4,5} for {Wake, N1, N2, N3, REM} 
%%
% should replace with your data path
raw_data_path = '/media/slow-storage/DataFolder/Kaare/packed_80/';

% save path for mat file
mat_path = './mat/';
if(~exist(mat_path, 'dir'))
    mkdir(mat_path);
end

fs = 100; % sampling frequency
win_size  = 2;  % window size for short-time Fourier Transform
overlap = 1;    % overlap size for short-time Fourier Transform
nfft = 2^nextpow2(win_size*fs);

listing = dir([raw_data_path, '*.mat']);

for i = 1 : numel(listing)
	disp(listing(i).name)
    
    load([raw_data_path, listing(i).name]);
    [~, filename, ~] = fileparts(listing(i).name);

    % label and one-hot encoding
    y = double(labels);
    label = zeros(size(y,1),1);
    for k = 1 : size(y,1)
        [~, label(k)] = find(y(k,:));
    end
    clear labels
    
    %% EEG
    N = size(data, 1);
    % raw input
    X1 = squeeze(data(:,:,1)); % raw input (Nx3000)
    % time-frequency input
    X2 = zeros(N, 29, nfft/2+1); % 29: the number windows in a 30-second epoch with the window size of 2s and 1s overlap
    for k = 1 : size(X1, 1)
        if(mod(k,100) == 0)
            disp([num2str(k),'/',num2str(N)]);
        end
        [Xk,~,~] = spectrogram(X1(k,:), hamming(win_size*fs), overlap*fs, nfft);
        % log magnitude spectrum
        Xk = 20*log10(abs(Xk));
        X2(k,:,:) = Xk';
    end
    X1 = single(X1);
    X2 = single(X2);
    y = single(y);
    label=single(label);
    save([mat_path, filename,'_eeg.mat'], 'X1', 'X2', 'label', 'y', '-v7.3');
    clear X1 X2
    
    %% EOG
    % raw input
    X1 = squeeze(data(:,:,2)); % raw input (Nx3000)
    % time-frequency input
    X2= zeros(N, 29, nfft/2+1); % 29: the number windows in a 30-second epoch with the window size of 2s and 1s overlap
    for k = 1 : size(X1, 1)
        if(mod(k,100) == 0)
            disp([num2str(k),'/',num2str(N)]);
        end
        [Xk,~,~] = spectrogram(X1(k,:), hamming(win_size*fs), overlap*fs, nfft);
        % log magnitude spectrum
        Xk = 20*log10(abs(Xk));
        X2(k,:,:) = Xk';
    end
    X1 = single(X1);
    X2 = single(X2);
    save([mat_path, filename,'_eog.mat'], 'X1', 'X2', 'label', 'y', '-v7.3');
    clear X1 X2
    
    %% EMG
    % raw input
    X1 = squeeze(data(:,:,3)); % raw input (Nx3000)
    % time-frequency input
    X2= zeros(N, 29, nfft/2+1);
    for k = 1 : size(eeg_epochs, 1)
        if(mod(k,100) == 0)
            disp([num2str(k),'/',num2str(N)]);
        end
        [Xk,~,~] = spectrogram(X1(k,:), hamming(win_size*fs), overlap*fs, nfft);
        % log magnitude spectrum
        Xk = 20*log10(abs(Xk));
        X2(k,:,:) = Xk';
    end
    X1 = single(X1);
    X2 = single(X2);
    save([mat_path, filename,'_emg.mat'], 'X1', 'X2', 'label', 'y', '-v7.3');
    clear X1 X2 y label
end