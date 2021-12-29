clear all
close all
clc


data_path = '/media/Dataset/SleepEDF-Expanded/sleep-cassette/';

raw_data_path = './raw_data_30min/';
if(~exist(raw_data_path, 'dir'))
    mkdir(raw_data_path);
end

% Sampling rate of hypnogram (default is 30s)
epoch_time = 30;
fs = 100; % sampling frequency

% list only healthy (SC) subjects
listing = dir([data_path, 'SC4*']);

for i = 1 : numel(listing)
    disp(listing(i).name)
    target_dir = [data_path, listing(i).name, '/'];
    
    [~,filename,~] = fileparts(listing(i).name);
    [sub_id,night] = edfx_dir2sub(filename);
    sub_id = sub_id + 1; % index 0 to 1
    
    % load edf data to get Fpz-Cz, and EOGhorizontal channels
    edf_file = [target_dir, listing(i).name, '-PSG.edf'];
    [header, edf] = edfreadUntilDone(edf_file);
    channel_names = header.label;
    
    for c = 1 : numel(channel_names)
        channel_names{c} = strtrim(channel_names{c});
    end
    chan_ind_eeg = find(ismember(channel_names, 'EEGFpzCz'));
    if(isempty(chan_ind_eeg))
        disp('Oops, wait. Channel not found!');
        pause;
    end
    
    if(header.frequency(chan_ind_eeg) ~= fs)
        disp('Oops, wait! Sampling frequency mismatched!');
        pause;
    end
    
    chan_ind_eog = find(ismember(channel_names, 'EOGhorizontal'));
    if(isempty(chan_ind_eog))
        disp('Oops, wait. Channel not found!');
        pause;
    end
    
    if(header.frequency(chan_ind_eog) ~= fs)
        disp('Oops, wait! Sampling frequency mismatched!');
        pause;
    end
    
    chan_data_eeg = edf(chan_ind_eeg, :);
    chan_data_eog = edf(chan_ind_eog, :);
    clear edf header channel_names
    
    % ensure the signal is calibrated to microvolts
    if(max(chan_data_eeg) <= 10)
        disp('Signal calibrated!');
        chan_data_eeg = chan_data_eeg * 1000;
    end
    
    assert(sum(isnan(chan_data_eeg)) == 0)
    assert(sum(isnan(chan_data_eog)) == 0)
    
    
    %% Preprocessing Filter coefficiens
    Nfir = 100;

    % Preprocessing filters
    b_band = fir1(Nfir,[0.3 40].*2/fs,'bandpass'); % bandpass
    chan_data_eeg = filtfilt(b_band,1,chan_data_eeg);

    % Preprocessing filters
    b_band = fir1(Nfir,[0.3 40].*2/fs,'bandpass'); % bandpass
    chan_data_eog = filtfilt(b_band,1,chan_data_eog);
    
    
    % load hypnogram
    hyp_file = fullfile([target_dir, 'info/', listing(i).name, '.txt']);
    hypnogram = edfx_load_hypnogram_new( hyp_file );
    disp(['Hypnogram length: ', num2str(length(hypnogram))]);
    
    % process times to determine the in-bed duration
    [chan_data_eeg_new, chan_data_eog_new, hypnogram_new] = edfx_process_time_2chan_moreW(target_dir, chan_data_eeg, chan_data_eog, hypnogram, epoch_time, fs);
    disp(['Number of epochs: ', num2str(length(hypnogram_new))]);
    
    eeg_epochs = buffer(chan_data_eeg_new, epoch_time*fs);
    eeg_epochs = eeg_epochs';
    
    eog_epochs = buffer(chan_data_eog_new, epoch_time*fs);
    eog_epochs = eog_epochs';

    label = edfx_hypnogram2label(hypnogram_new);
    
    assert(size(eeg_epochs,1) == numel(label));
    
    % excluding Unknown and non-score
    ind = (label == 0);
    disp([num2str(sum(ind)), ' epochs excluded.'])
    label(ind) = [];
    eeg_epochs(ind,:,:) = [];
    eog_epochs(ind,:,:) = [];

    data = zeros([size(eeg_epochs), 2]);
    data(:,:,1) = eeg_epochs;
    data(:,:,2) = eog_epochs;
    
    % since the d
    y = zeros(numel(label),1);
    for k = 1 : numel(label)
        y(k, label(k)) = 1;
    end
    
    save([raw_data_path, 'n', num2str(sub_id,'%02d'), '_', num2str(night), '.mat'], 'data', 'label', 'y', '-v7.3');
    clear data label y
end
