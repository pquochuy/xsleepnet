function [subject, night] = edfx_dir2sub(target_dir)
    subject = str2double(target_dir(4:5));
    night = str2double(target_dir(6));
end