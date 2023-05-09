%% Question 2 part c

clear;
close all;
clc;

city_data = load ('city_data.mat');
city_data = city_data.city_data;

figure
scatter(city_data(:,1),city_data(:,2),'o','filled');
title('City-Data-Scatterplot');
grid on

%%%%%% Creating the graph
weight = zeros(1,50);
for i = 1:50
    for j = 1:50
        weight(i,j) = norm(city_data(i,:)-city_data(j,:));
    end
end


