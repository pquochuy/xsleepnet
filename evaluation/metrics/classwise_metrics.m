function [f1, sens, spec] = classwise_metrics(y,yhat,class)
    ind = (y == class);
    y(~ind) = 0;
    y(ind) = 1;
    
    ind = (yhat == class);
    yhat(~ind) = 0;
    yhat(ind) = 1;

    bin_metrics = binary_metrics(y,yhat);
    f1 = bin_metrics(6);
    sens = bin_metrics(2);
    spec = bin_metrics(3);
end