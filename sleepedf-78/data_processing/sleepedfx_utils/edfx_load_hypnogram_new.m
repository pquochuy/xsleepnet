function [hypnogram] = edfx_load_hypnogram_new( hyp_file )
%processEDFxHypnogram Reads the annotation file to produce a hypnogram
%   [hypnogram] = processEDFxHypnogram(hyp_file) uses the annotation file
%   downloaded to produce a per-epoch hypnogram with the following labels:
%   W, 1, 2, 3, 4, R, M


table = readtable(hyp_file, 'Delimiter','tab');
stages = table{:,3};
durations = table{:,2};
clear table

epoch_size = 30;

% Total (round) number of epochs
number_of_epochs = sum(durations)/epoch_size;

% Container for hypnogram
hypnogram = char.empty(number_of_epochs,0);

% Using the duration of each stage, assign hypnogram value for each 30s
% epoch in the vector
idx=0;
for h=1:length(durations)
    ep_end = durations(h)/epoch_size;
    if(strcmp(stages{h}, 'Movement time'))
        hypnogram(idx+1:idx+ep_end,1)='M';
    else
        hypnogram(idx+1:idx+ep_end,1)=stages{h}(end);
    end
    idx=idx+ep_end;
end

%{
% Convert to AASM if that is the classification_mode
% Conversion is as follows
% M  -> W
% 4  -> 3
% Rest are same: W,1,2,3,R
if strcmp(classification_mode,'AASM')
    hypnogram(hypnogram=='M')='W';
    hypnogram(hypnogram=='4')='3';
end
%}
end