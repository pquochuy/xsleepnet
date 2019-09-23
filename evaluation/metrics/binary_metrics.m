function bin_metrics = binary_metrics(y,yhat)
    idx = (y()==1);

    p = length(y(idx));
    n = length(y(~idx));
    N = p+n;

    tp = sum(y(idx)==yhat(idx));
    tn = sum(y(~idx)==yhat(~idx));
    fp = n-tn;
    fn = p-tp;

    tp_rate = tp/p;
    tn_rate = tn/n;

    accuracy = (tp+tn)/N;
    sensitivity = tp_rate;
    specificity = tn_rate;
    precision = tp/(tp+fp);
    recall = sensitivity;
    f_measure = 2*((precision*recall)/(precision + recall));
    gmean = sqrt(tp_rate*tn_rate);

    bin_metrics = [accuracy sensitivity specificity precision recall f_measure gmean];
end