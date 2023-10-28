function obj = SVM(var)
% add path
    addpath(genpath('utils'));
    addpath(genpath('libsvm-3.21'));
    addpath(genpath('liblinear-2.1'));
    
    accuracy = [];
    % load data set
    [train_LABEL, train_DATA] = libsvmread('datasets/pendigits'); 
    [test_LABEL, test_DATA] = libsvmread('datasets/pendigits.t');


    if min(train_LABEL) ==0
        train_LABEL = train_LABEL+1;
        test_LABEL = test_LABEL+1;
    end
    class_num = max(train_LABEL)-min(train_LABEL)+1;
    seg_N = floor(size(test_LABEL,1)/3);
    
    train_label = train_LABEL;
    train_data = train_DATA;
    val_label = test_LABEL(1:seg_N);
    val_data = test_DATA(1:seg_N,:);   
    
        
    linear_model =[];
    % Train Linear Models with Liblinear
    label_sets = nchoosek(min(train_label):max(train_label),2);
    for i = 1:size(label_sets,1)
        class = label_sets(i,:);

        % 1. Prepare data matrix
        tmp_train = find(train_label==class(1)|train_label==class(2));
        tr_data = train_data(tmp_train,:);
        tr_label = train_label(tmp_train,:);
        tr_label(tr_label==class(1)) = -1;
        tr_label(tr_label==class(2)) = 1;

        % Use linear kernel for large data set
        arg1 = strcat({'-s 1 -c '}, num2str(var(i)), ' -q');
        linear_model = [linear_model, train(tr_label, tr_data, arg1{1})];
    end

    true_label = 0;
    for i = 1:size(val_label,1)
        result = zeros(1,class_num);
        for j = 1:size(label_sets,1)
            class = label_sets(j,:);
            [linear_label, linear_accuracy, v] = predict(val_label(i), val_data(i,:), linear_model(j), '-q');
            if(linear_label==-1)
                result(class(1)) = result(class(1))+1;
            else
                result(class(2)) = result(class(2))+1;
            end
        end
        [a,tmp_label] = max(result);
        if val_label(i) == tmp_label
            true_label = true_label+1;
        end
    end
    accuracy  = [accuracy, true_label/ size(val_label,1)];
    obj = 1 - accuracy;
%     disp(obj);
end
