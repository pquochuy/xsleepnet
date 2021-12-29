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

% list all subjects
listing = dir([raw_data_path, 'SS*']);

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
    X1 = squeeze(data(:,:,1)); % eeg channel
    X2 = zeros(N, 29, nfft/2+1);
    eeg_epochs = squeeze(data(:,:,1)); % eeg channel
    for k = 1 : size(eeg_epochs, 1)
        if(mod(k,100) == 0)
            disp([num2str(k),'/',num2str(size(eeg_epochs, 1))]);
        end
        [Xk,~,~] = spectrogram(eeg_epochs(k,:), hamming(win_size*fs), overlap*fs, nfft);
        % log magnitude spectrum
        Xk = 20*log10(abs(Xk));
        Xk = Xk';
        X2(k,:,:) = Xk;
    end
    X1 = single(X1);
    X2 = single(X2);
    y = single(y);
    label=single(label);
    save([mat_path, filename,'_eeg.mat'], 'X1', 'X2', 'label', 'y', '-v7.3');
    clear X1 X2
    
    %% EOG
    X1 = squeeze(data(:,:,2)); % eog channel
    X2= zeros(N, 29, nfft/2+1);
    eeg_epochs = squeeze(data(:,:,2)); % eog channel
    for k = 1 : size(eeg_epochs, 1)
        if(mod(k,100) == 0)
            disp([num2str(k),'/',num2str(size(eeg_epochs, 1))]);
        end
        [Xk,~,~] = spectrogram(eeg_epochs(k,:), hamming(win_size*fs), overlap*fs, nfft);
        % log magnitude spectrum
        Xk = 20*log10(abs(Xk));
        Xk = Xk';
        X2(k,:,:) = Xk;
    end
    X1 = single(X1);
    X2 = single(X2);
    y = single(y);
    label=single(label);
    save([mat_path, filename,'_eog.mat'], 'X1', 'X2', 'label', 'y', '-v7.3');
    clear X1 X2
    
    %% EMG
    X1 = squeeze(data(:,:,3)); % eog channel
    X2= zeros(N, 29, nfft/2+1);
    eeg_epochs = squeeze(data(:,:,3)); % emg channel
    for k = 1 : size(eeg_epochs, 1)
        if(mod(k,100) == 0)
            disp([num2str(k),'/',num2str(size(eeg_epochs, 1))]);
        end
        [Xk,~,~] = spectrogram(eeg_epochs(k,:), hamming(win_size*fs), overlap*fs, nfft);
        % log magnitude spectrum
        Xk = 20*log10(abs(Xk));
        Xk = Xk';
        X2(k,:,:) = Xk;
    end
    X1 = single(X1);
    X2 = single(X2);
    y = single(y);
    label=single(label);
    save([mat_path, filename,'_emg.mat'], 'X1', 'X2', 'label', 'y', '-v7.3');
    clear X1 X2 y label
end