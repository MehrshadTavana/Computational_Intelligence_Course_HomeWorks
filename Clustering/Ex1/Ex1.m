%% Question 1 Part a

clc;
clear;
close all;

data = load('SampleData1.mat');
data_training = data.TrainingData;
data_label = data.TrainingLabels;

data_training_class0 = data_training(:,(data_label == 0));
data_training_class1 = data_training(:,(data_label == 1));

figure
scatter(data_training_class0(1,:),data_training_class0(2,:),'b','filled');
hold on 
scatter(data_training_class1(1,:),data_training_class1(2,:),'r','filled');
grid on
legend('Class 0','Class 1');
xlabel('dim1')
ylabel('dim2')
title('scatterplot of data')


%% Question 1 Part b
%%%%% I prefered to select first 70% of data as training data and the rest as
%%%%% the validation data so we have:

validation_index = (281:400);
validation_data = data_training(:,validation_index);
validation_label = data_label(validation_index);

training_index = (1:280);
training_data = data_training(:,training_index);
training_label = data_label(training_index);


%% Question 1 Part c
n = 50;
index = (1:n);
performance = zeros(1,n);
temp_pref = zeros(1,3);
error = zeros(1,n);

for i = 1:n
    hidden_layer_size = i;
    
    for j = 1:3
    net = feedforwardnet(hidden_layer_size);
    net.layers{1}.transferFcn = 'tansig'; 
    net = train(net,training_data,training_label);
    y_validation = net(validation_data);
    lables_prediction = (validation_data> 0);
    performance = perform(net,validation_label,y_validation);
    temp_pref(j) = performance;
    end
    error(i) = mean(temp_pref);
end

[~,num]=min(error);
figure
bar(1:n,error);
title('Error Plot')
xlabel('number of neurons of hidden layer')
ylabel('error')
fprintf(['Best number of neurons of hidden layer = ',num2str(index(num)),'\n']);


%% Question 1 Part d
%%%%% Caution!! This part may take a some time to run completely
m = 50;
radius = [0.1,0.2,0.5,1,2,4,8,10];
layers = (1:m);
error_rbf = zeros(length(radius),length(layers));

for i=1:length(radius)
    for j=1:length(layers)
        net = newrb(training_data,training_label,0,radius(i),layers(j));
        y_validation_rbf = net(validation_data);
        performace = perform(net,validation_label,y_validation_rbf);
        error_rbf(i,j) = performace;
    end
end

[num1,num2] = min(error_rbf(:));
[radius_index, layers_index] = ind2sub(size(error_rbf),num2);

fprintf(['Best number of neurons of hidden layer: ',num2str(layers(layers_index)),'\n'])
fprintf(['Best radius for our RBF NN: ',num2str(radius(radius_index)),'\n'])
