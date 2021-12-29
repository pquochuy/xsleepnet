clear all
close all
clc

raw_data_path = './raw_data/';
mat_path = './mat/';
if(~exist(mat_path, 'dir'))
    mkdir(mat_path);
end

fs = 100; % sampling frequency

win_size  = 2;
overlap = 1;
nfft = 2^nextpow2(win_size*fs);

list = dir([raw_data_path, '*.mat']);

count = 0;
cur_sub = -1;
for i = 1 : numel(list)
    sub = str2num(list(i).name(2:3));
    night = str2num(list(i).name(5));
    if(sub ~= cur_sub)
        cur_sub = sub;
        count = count + 1;
    end
    load([raw_data_path, list(i).name]);
    
    %% EEG
    X1 = data(:,:,1);
    % short time fourier transform
    N = size(X1, 1);
    X2 = zeros(N, 29, nfft/2+1);
    for k = 1 : size(X1, 1)
        [Xk,~,~] = spectrogram(X1(k,:), hamming(win_size*fs), overlap*fs, nfft);
        % log magnitude spectrum
        Xk = 20*log10(abs(Xk));
        Xk = Xk';
        X2(k,:,:) = Xk;
    end
    y = single(y); % one-hot encoding
    label = single(label);
    X2 = single(X2);
    X1 = single(X1);

    save([mat_path, 'n', num2str(count,'%02d'), '_', num2str(night),'_eeg.mat'], 'X1', 'X2', 'label', 'y', '-v7.3');
    clear X1 X2 
    
    %% EOG
    X1 = data(:,:,2);
    % short time fourier transform
    N = size(X1, 1);
    X2 = zeros(N, 29, nfft/2+1);
    for k = 1 : size(X1, 1)
        [Xk,~,~] = spectrogram(X1(k,:), hamming(win_size*fs), overlap*fs, nfft);
        % log magnitude spectrum
        Xk = 20*log10(abs(Xk));
        Xk = Xk';
        X2(k,:,:) = Xk;
    end
    y = single(y); % one-hot encoding
    label = single(label);
    X2 = single(X2);
    X1 = single(X1);

    save([mat_path, 'n', num2str(count,'%02d'), '_', num2str(night),'_eog.mat'], 'X1', 'X2', 'label', 'y', '-v7.3');
end
