%%%%%%%%%% Ahmadreza Tavana Ex2 98104852  

%% Part a
clc;
clear;
close all;

data2 = load('Ex2');

train_data_original = data2.TrainData;
train_data = data2.TrainData(:,1:72);
validation_data = data2.TrainData(:,73:end);

train_data_original_0 = train_data_original(:,train_data_original(4,:) == 0);
train_data_original_1 = train_data_original(:,train_data_original(4,:) == 1);

test_data = data2.TestData;

%%%%% plotting data
figure;
scatter3(train_data_original_0(1,:),train_data_original_0(2,:),train_data_original_0(3,:),...
    'filled','b');
hold on
scatter3(train_data_original_1(1,:),train_data_original_1(2,:),train_data_original_1(3,:),...
    'filled','r');
title('Scatter Plot')
legend('Class 0','Class 1')


%%%% training NN
train_data(4,:) = (2*train_data(4,:))-1;
accuracy_a = zeros(1,10);

for i = 1:50 
    hidden_layer_size = i;
    net = feedforwardnet(hidden_layer_size);
    net.layers{1}.transferFcn = 'tansig'; 
    net = train(net,train_data(1:3,:),train_data(4,:));
    y_validation = net(validation_data(1:3,:));
    lables_prediction_a = (y_validation> 0);
    size_validation = size(validation_data(4,:));
    accuracy_a(i) = sum((lables_prediction_a == validation_data(4,:)) == 1)./(size_validation(2));
end

figure
bar(1:50,accuracy_a)
title('accuracy-a')
xlabel('number of neurons of hidden layer')
ylabel('accuracy')


y_test = net(test_data);
labels_a = (y_test> 0);
save('Testlabel_a.mat','labels_a');


%% part b
train_labels = [train_data(4,:); -train_data(4,:)];
validation_labels = [validation_data(4,:); -validation_data(4,:)];
accuracy_b = zeros(1,10);

for i = 1:50
    hidden_layer_size = i;
    net = feedforwardnet(hidden_layer_size);
    net.layers{1}.transferFcn = 'tansig'; 
    net = train(net,train_data(1:3,:),train_labels);
    y_validation = net(validation_data(1:3,:));
    lables_prediction_b = (y_validation(1,:)> y_validation(2,:));
    size_validation = size(validation_data(4,:));
    accuracy_b(i) = sum((lables_prediction_b == validation_data(4,:)) == 1)./(size_validation(2));
end

figure
bar(1:50,accuracy_b)
title('accuracy-b')
xlabel('number of neurons of hidden layer')
ylabel('accuracy')


y_test = net(test_data);
labels_b = y_test(1,:) > y_test(2,:);

save('Testlabel_b.mat','labels_b');
