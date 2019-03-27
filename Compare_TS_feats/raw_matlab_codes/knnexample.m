

% KNN on feature vectors
% Take one label and feature vector out
% Use the rest of the dataset (71) labels to try and classify it
% as 1 of 12 possible labels

%{
VOC_inds = []; 
for i = 1:5
    VOC_inds = [VOC_inds, 219*3*(i-1) + (1:219)];
end

P_inds = []; 
for i = 1:5
    P_inds = [P_inds, 219 + 219*3*(i-1) + (1:219)];
end

T_inds = []; 
for i = 1:5
    T_inds = [T_inds, 2*219 + 219*3*(i-1) + (1:219)];
end
%}
n = size(lv_norm);
KNN_K = 3;
num_vecs = n(1);

for j = 1:n(2)
    feat_matrix = lv_norm(:,j);

    for i = 1:num_vecs
        % one_out(i) = randsample(72,1);
        one_out(i) = i;

        % Assign training feature matrix
        lv_norm_train = feat_matrix;
        lv_norm_train(one_out(i),:) = [];

        % Assign training feature labels
        labels_train = labels;
        labels_train(one_out(i)) = [];

        % Assign test feature matrix and labels
        lv_norm_test = feat_matrix(one_out(i),:);
        labels_test(i) = labels(one_out(i));

        % Model KNN and predict the class of the test feature vector

        Mdl = fitcknn(lv_norm_train,labels_train,'NumNeighbors',KNN_K);
        pred_class(i) = predict(Mdl,lv_norm_test);

        % accuracy_s(i) = sum(pred_class == labels_test)/num_vecs;
        
    end
    
    accuracy(j) = sum(pred_class == labels_test)/length(pred_class);
    disp([num2str(j),': ',num2str(accuracy(j))])
    
end


figure
counter = 0;
for i = 1
    for j = 1:3
        counter = counter + 1;
        subplot(1,3,counter)
        stem(accuracy((counter-1)*219+(1:219)),'.')
        ylim([0 1])
    end
end
    






