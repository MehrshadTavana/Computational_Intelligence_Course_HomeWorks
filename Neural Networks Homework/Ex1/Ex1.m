%%%%%%%%%% Ahmadreza Tavana Question 1 98104852

%% Part a
clc;
clear;
close all;

data1 = load('Ex1.mat');
NOemission = data1.NOemission;
fuelrate = data1.fuelrate;
speed = data1.speed;

figure();

scatter3(speed,NOemission,fuelrate)

title('Scatter plot of fuerate by Noemission and speed');
xlabel('speed');
ylabel('NOemission');
zlabel('fuelrate');



%% Part b and c
clc
close all

train_data_speed = speed(1:700);
train_data_NOemission = NOemission(1:700);
train_data_fuelrate = fuelrate(1:700);

validation_data_speed = speed(701:end);
validation_data_NOemission = NOemission(701:end);
validation_data_fuelrate = fuelrate(701:end);

train_data = [train_data_speed; train_data_NOemission; train_data_fuelrate]';
validation_data = [validation_data_speed; validation_data_NOemission; validation_data_fuelrate]';
    
fit = fitlm(train_data(:,1:2),train_data(:,3));
display(fit)

coefficient_estimation = fit.Coefficients.Estimate;
fuelrate_estimation_train = coefficient_estimation(1)....
                          + coefficient_estimation(2)*train_data_speed...
                          + coefficient_estimation(3)*train_data_NOemission;
                  
MSE_tarin_data = mean((train_data_fuelrate - fuelrate_estimation_train).^2);
display(MSE_tarin_data)

fuelrate_estimation_validation = coefficient_estimation(1)...
                               + coefficient_estimation(2)*validation_data_speed...
                               + coefficient_estimation(3)*validation_data_NOemission;
                  
MSE_validation_data = mean((validation_data_fuelrate - fuelrate_estimation_validation).^2);
display(MSE_validation_data)




%% part d

close all;

Y = max(train_data_fuelrate) + 1;
y = train_data_fuelrate;

y_new = log((Y-y)./y);

fit_log = fitlm(train_data(:,1:2),y_new);
display(fit_log)

coefficient_estimation_log = fit_log.Coefficients.Estimate;

fuelrate_estimation_train_log = coefficient_estimation_log(1)....
                          + coefficient_estimation_log(2)*train_data_speed...
                          + coefficient_estimation_log(3)*train_data_NOemission;
                  
MSE_train_data_log = mean((train_data_fuelrate - fuelrate_estimation_train_log).^2);
display(MSE_train_data_log)

fuelrate_estimation_validation_log = coefficient_estimation_log(1)...
                               + coefficient_estimation_log(2)*validation_data_speed...
                               + coefficient_estimation_log(3)*validation_data_NOemission;
                  
MSE_validation_data_log = mean((validation_data_fuelrate - fuelrate_estimation_validation_log).^2);
display(MSE_validation_data_log)


%% part e
hidden_layer = 100;
net = fitnet(hidden_layer);
net = train(net,train_data(:,1:2)',train_data(:,3)');

y_train = net(train_data(:,1:2)');
MSE_train_nn = mean((train_data(:,3)-y_train').^2);

y_validation = net(validation_data(:,1:2)');
MSE_valid_nn = mean((validation_data(:,3)-y_validation').^2);