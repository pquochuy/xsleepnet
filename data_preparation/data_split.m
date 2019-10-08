clear all
close all
clc


rng(10); % for repeatable

% divide MASS subjects into training, evaluation, and test sets

Nsub = 200;

subjects = randperm(Nsub);

Nfold = 20;
Ntest = Nsub/Nfold;

test_sub = cell(Nfold,1);
eval_sub = cell(Nfold,1);
train_sub = cell(Nfold,1);


for s = 1 : Nfold
    test_s = subjects((s-1)*Ntest + 1:s*Ntest);
    
    rest_s = setdiff(subjects, test_s); 
    perm_list = randperm(numel(rest_s));
    
    % 10 subjects as eval set
    eval_s = sort(rest_s(perm_list(1:10)));
    train_s = sort(rest_s(perm_list(11:end)));
    
    test_sub{s} = sort(test_s);
    train_sub{s} = sort(train_s);
    eval_sub{s} = sort(eval_s);
end

save('./data_split.mat', 'train_sub','test_sub','eval_sub');