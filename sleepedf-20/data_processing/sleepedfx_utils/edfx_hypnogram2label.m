function [label] = edfx_hypnogram2label(hypnogram)
    label =zeros(numel(hypnogram), 1);
    % Convert to AASM if that is the classification_mode
    % Conversion is as follows
    % M  -> W
    % 4  -> 3
    % Rest are same: W,1,2,3,R
    %hypnogram(hypnogram_f=='M')='W'; % exclude movement
    hypnogram(hypnogram=='4')='3';
    target_label = ['W','1','2','3','R'];
    %target_label = ['W','R','1','2','3']; % reorder to be compatible with MASS database
    for i = 1 : numel(target_label)
        label(hypnogram == target_label(i)) = i;
    end
end