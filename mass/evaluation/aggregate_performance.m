% net_name: 'xsleepnet2', 'xsleepnet1', 'naive-fusion', 'seqsleepnet', 'fcnn-rnn'
function [acc, kappa, f1, sens, spec, classwise_sens, classwise_sel, C] = aggregate_performance(net_name, nchan)

    Nfold = 20;
    yh = cell(Nfold,1);
    yt = cell(Nfold,1);
    mat_path = './mat/';
    listing = dir([mat_path, '*_eeg.mat']);
    load('./data_split_eval.mat');

    seq_len = 20;

    for fold = 1 : Nfold
        test_s = test_sub{fold};
        sample_size = zeros(numel(test_s), 1);
        for i = 1 : numel(test_s)
            sname = listing(test_s(i)).name;
            load([mat_path,sname], 'label');
            % handle the different here
            sample_size(i) = numel(label) -  (seq_len - 1); 
            yt{fold} = [yt{fold}; double(label)];
        end
        if ismember(net_name, {'xsleepnet1', 'xsleepnet2'})
            load(['./tensorflow_nets/', net_name, '/scratch_training_',num2str(nchan),'chan/n',num2str(fold),'/test_ret_joint.mat']);
        else
            load(['./tensorflow_nets/', net_name, '/scratch_training_',num2str(nchan),'chan/n',num2str(fold),'/test_ret.mat']);
        end
        
        score_ = cell(1,seq_len);
        for n = 1 : seq_len
            if ismember(net_name, {'xsleepnet1', 'xsleepnet2'})
                score_{n} = softmax(squeeze(joint_score(n,:,:)));
            else
                score_{n} = softmax(squeeze(score(n,:,:)));
            end
        end
        score = score_;
        clear score_;

        for i = 1 : numel(test_s)
            start_pos = sum(sample_size(1:i-1)) + 1;
            end_pos = sum(sample_size(1:i-1)) + sample_size(i);
            score_i = cell(1,seq_len);
            for n = 1 : seq_len
                score_i{n} = score{n}(start_pos:end_pos, :);
                N = size(score_i{n},1);

                score_i{n} = [ones(seq_len-1,5); score{n}(start_pos:end_pos, :)];
                score_i{n} = circshift(score_i{n}, -(seq_len - n), 1);
            end

            fused_score = log(score_i{1});
            for n = 2 : seq_len
                fused_score = fused_score + log(score_i{n});
            end

            yhat = zeros(1,size(fused_score,1));
            for k = 1 : size(fused_score,1)
                [~, yhat(k)] = max(fused_score(k,:));
            end

            yh{fold} = [yh{fold}; double(yhat')];
        end
    end

    yh = cell2mat(yh);
    yt = cell2mat(yt);
    
    [acc, kappa, f1, sens, spec] = calculate_overall_metrics(yt, yh);
    [classwise_sens, classwise_sel]  = calculate_classwise_sens_sel(yt, yh);
    C = confusionmat(yt, yh);
end

