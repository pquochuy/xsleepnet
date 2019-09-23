function [acc, kappa, f1, sens, spec] = calculate_overall_metrics(y, yhat)
    
    Ncat = numel(unique(y));
    
    acc = sum(y == yhat)/numel(yhat);
    kappa = cohensKappa(yhat,y);
    
    f1 = zeros(Ncat,1);
    sens = zeros(Ncat,1);
    spec = zeros(Ncat,1);
    for cl = 1 : Ncat
        [f1(cl), sens(cl), spec(cl)]  = classwise_metrics(y,yhat,cl);
    end
    f1 = mean(f1);
    sens = mean(sens);
    spec = mean(spec);
end