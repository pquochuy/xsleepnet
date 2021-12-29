function [chan_data_eeg_new, chan_data_eog_new, hypnogram_new] = edfx_process_time_2chan(target_dir, chan_data_eeg, chan_data_eog, hypnogram, epoch_time, fs)
    info_dir = [target_dir, 'info/'];
    
    % Load all time values from text files
    %lights_off_time = textread(fullfile(info_dir, 'lights_off_time.txt'),'%s');
    fid = fopen(fullfile(info_dir, 'lights_off_time.txt'));
    lights_off_time = textscan(fid,'%s');
    fclose(fid);
    lights_off_time = lights_off_time{1};
    
    %rec_start_time = textread(fullfile(info_dir, 'rec_start_time.txt'),'%s');
    fid = fopen(fullfile(info_dir, 'rec_start_time.txt'));
    rec_start_time = textscan(fid,'%s');
    fclose(fid);
    rec_start_time = rec_start_time{1};
    
    %hyp_start_time  = textread(fullfile(info_dir, 'hyp_start_time.txt'),'%s');
    fid = fopen(fullfile(info_dir, 'hyp_start_time.txt'));
    hyp_start_time = textscan(fid,'%s');
    fclose(fid);
    hyp_start_time  = hyp_start_time{1};
    
    %lights_on_time   = textread(fullfile(info_dir, 'lights_on_time.txt'),'%s');
    fid = fopen(fullfile(info_dir, 'lights_on_time.txt'));
    lights_on_time = textscan(fid,'%s');
    fclose(fid);
    lights_on_time   = lights_on_time{1};
    
    %rec_stop_time   = textread(fullfile(info_dir, 'rec_stop_time.txt'),'%s');
    fid = fopen(fullfile(info_dir, 'rec_stop_time.txt'));
    rec_stop_time = textscan(fid,'%s');
    fclose(fid);
    rec_stop_time   = rec_stop_time{1};
    
    % Convert the times to a date vector
    lights_off_vec = datevec(lights_off_time);
    %lights_off_vec = datevec(addtodate(datenum(lights_off_time),-30,'minute'));
    rec_start_vec = datevec(rec_start_time);
    lights_on_vec = datevec(lights_on_time);
    %lights_on_vec = datevec(addtodate(datenum(lights_on_time),30,'minute'));
    hyp_start_vec = datevec(hyp_start_time);
    rec_stop_vec = datevec(rec_stop_time);
    
    % Check if hyp_start_time and rec_start_time are different and of different
    % days (i.e. past midnight)
    hs_flag = ~(sum(hyp_start_vec==rec_start_vec)==6); 
    hs_diff = etime(hyp_start_vec,rec_start_vec);
    if hs_flag
        if hs_diff < 0
            hyp_start_vec(3)=2;
            hs_diff = etime(hyp_start_vec,rec_start_vec);
        end
    end

    % Check if lights on and recording start time are same day or different
    et_diff = etime(lights_on_vec,rec_start_vec);
    if et_diff < 0
        lights_on_vec(3)=2;
        et_diff = etime(lights_on_vec,rec_start_vec);
    end
    rec_stop_vec(3)=lights_on_vec(3);
    
    % Check if lights off and recording start time are same day or different
    lo_diff = etime(lights_off_vec,rec_start_vec);
    if lo_diff < 0 && lights_off_vec(4)-rec_start_vec(4)<0
        lights_off_vec(3)=2;
        lo_diff = etime(lights_off_vec,rec_start_vec);
    end

    % At this point all the dates have been corrected for
    % the next step is to choose either hyp_start of lights_off
    % as the begin time


    % Determine which is the latest time to use as the begin time 
    % from which to read data from
    bt_diff = etime(lights_off_vec,hyp_start_vec);
    if bt_diff > 0
        begin_time = lights_off_time;
        btvec = lights_off_vec;
    else
        begin_time = hyp_start_time;
        btvec = hyp_start_vec;
    end
    
    % Difference between recording stop and lights on time to determine which
    % to use as the end time
    end_time_diff = etime(rec_stop_vec,lights_on_vec);
    if end_time_diff < 0
        ftvec = rec_stop_vec;
    else
        ftvec = lights_on_vec;
    end
    
    % Duration of time between these times
    data_duration = etime(ftvec, btvec);
    % Number of epochs obtained from this duration
    epochs_from_duration = floor(data_duration / epoch_time);
    
        % Select the right number of epochs to use in case the size of hypnogram
    % shows a different number of epochs
    if length(hypnogram)  < epochs_from_duration
        epochs_to_use = length(hypnogram);
    else
        epochs_to_use = epochs_from_duration;
    end
    
	% Index of start and end samples to read data
    start_eeg = etime(btvec,rec_start_vec) * fs + 1;
    end_eeg  = start_eeg + epochs_to_use * epoch_time * fs - 1;
    
    % truncate data to bed-time duration
    chan_data_eeg_new = chan_data_eeg(start_eeg : end_eeg);
    chan_data_eog_new = chan_data_eog(start_eeg : end_eeg);
    
    % Determine number of epochs
    number_of_epochs = length(chan_data_eeg_new)/(fs*epoch_time);
    
    % Find the hypnogram start index for slicing since that is not the same as
    % data start time or end time
    hyp_offset = etime(btvec, hyp_start_vec) / 30;
    if (hyp_offset < 0)
        error('ERROR: hyp_offset < 0');
    end
    
    % Hypnogram start index
    h_start = hyp_offset + 1;

    % Hypnogram end index
    h_end = hyp_offset + epochs_to_use;
    
    hypnogram_new = hypnogram(h_start:h_end);
end

