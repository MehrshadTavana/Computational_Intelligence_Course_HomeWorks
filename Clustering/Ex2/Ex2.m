%% Question 1 Part b and c

clc;
clear;
close all;

%%%%% at first I plot the data with scatter plot to have a sence of
%%%%% distribution of the data

data = load('SampleData2.mat');
data = data.DataNew;

figure
scatter(data(1,:),data(2,:),'filled')
xlabel('dim1')
ylabel('dim2')
title('scatter plot of data')
grid on

%%%% for choosing your considered k, please uncomment that
%k = 4;
k = 5;
%k = 6;
dist = size(data,1);

%%%% for choosing your considered center point, please uncomment that
center_points = rand(dist,k)*50-20;
%center_points = rand(dist,k)*70-30;
%center_points = rand(dist,k)*40-10;
%center_points = rand(dist,k)*90-50;

k_means_clustering = k_means(data,k,center_points);

figure

for i=1:k
    scatter(data(1,k_means_clustering==i),data(2,k_means_clustering==i),'filled')
    hold on;
    xlabel('dim1')
    ylabel('dim2')
    title(sprintf('Scatter plot of clusters (k-means) with k = %d',k))
    grid on;
end

%% Question 2 Part d

[groups,Center] = kmeans(data',k);

figure
gscatter(data(1,:),data(2,:),groups)
xlabel('dim1')
ylabel('dim2')
title(sprintf('Plot of clusters (k-means by MATLAB function) with k = %d',k))
grid on;

%% Question 2 Part e
%%%%% in this part I use hierarchical clustering with linkage and cluster
%%%%%(I explain the hierarchical clustering in my report for this part)
figure
Z = linkage(data','ward');
c = cluster(Z,'Maxclust',k);
scatter(data(1,:),data(2,:),30,c,'filled')
xlabel('dim1')
ylabel('dim2')
title(sprintf('Plot of clusters (hierarchical clustering by MATLAB function) with k = %d',k))
grid on;


%% Question 2 Part f

%%%%% at first I plot the data with scatter plot to have a sence of
%%%%% distribution of the data

data2 = load('SampleData3.mat');
data2 = data2.DataNew2;


figure
scatter(data2(1,:),data2(2,:),'filled')
xlabel('dim1')
ylabel('dim2')
title('scatter plot of data2')
grid on



%%%%%%%% k-means clustering for SampleData3
k = 3;
dist2 = size(data2,1);
center_points2 = rand(dist2,k)+2.5;

k_means_clustering = k_means(data2,k,center_points2);

figure

for i=1:k
    scatter(data2(1,k_means_clustering==i),data2(2,k_means_clustering==i),'filled')
    hold on;
    xlabel('dim1')
    ylabel('dim2')
    title(sprintf('Scatter plot of clusters (k-means) with k = %d',k))
    grid on;
end



%%%%%%% hierarchical clustering for SampleData3

figure
Z = linkage(data2','ward');
c = cluster(Z,'Maxclust',k);
scatter(data2(1,:),data2(2,:),35,c,'filled')
xlabel('dim1')
ylabel('dim2')
title(sprintf('Scatter plot of clusters (hierarchical clustering) with k = %d',k))
grid on;


