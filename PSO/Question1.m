%% Question 1 part a

clc
clear
close all

fx = @(x) -2*cos(3*x) + x.^2/8;
figure;
fplot(fx,[-2*pi 2*pi],'k');
title('y = f(x)');
xlabel('x');
ylabel('y');
grid on

%% Question 1 part b

iteration = 200;  %% select iteration 
ethaVec = [0.01, 0.05, 0.1, 0.2, 0.5, 1];  %% total ethas
result = zeros(1,6);
temp = 1;
fprintf('part b');
fprintf('\n');
for etha = ethaVec
    k = 0;
    for j = 1:1000
        [x] = Gradient_Descent(fx,-2*pi,2*pi,iteration,etha);  %% gradiant_descent
        if (abs(fx(x)+2) < 1e-4)
            k = k + 1;
        end
    end
    result(temp) = k;
   
    fprintf('The number of convergence for alpha = %d ',etha);
    fprintf('is equal to %d ',result(temp));
    fprintf('\n');
    temp = temp + 1;
end

fprintf('\n');

%% Question 1 part c

result = 0;
iteration = 200;

for i = 1:1000
    [x] = Newton_Rafson(fx,-2*pi,2*pi,iteration);
    if (abs(fx(x)+2) < 0.7)
        result = result + 1;
    end
end

fprintf('part c');
fprintf('\n');
fprintf('The number of convergence is equal to %d',result);
fprintf('\n')
fprintf('\n');


%% Question 1 part d

iteration = 200;
gama = 0.8;
alphaVec = [0.1, 0.5, 1, 2, 4, 6, 8,10];
result_Annual = zeros(1,8);
temp = 1;
fprintf('part d');
fprintf('\n');
for alpha = alphaVec
    for i = 1:1000
        [x] = Simulated_Annual(fx,-2*pi,2*pi,iteration,gama,alpha);
        if (abs(fx(x)+2) < 1e-2)
            result_Annual(temp) = result_Annual(temp) + 1;
        end
    end
    
    fprintf('The number of convergence for alpha = %d ',alpha);
    fprintf('is equal to %d ',result_Annual(temp));
    fprintf('\n');
    temp = temp + 1;
end

fprintf('\n');


%% Functions

function f_p = fx_derivative(fx,startingPoint,finishPoint,x)

    interval = finishPoint - startingPoint;
    h = interval * 10^-6;    
    f_p = (fx(x + h) - fx(x - h))/(2*h);
    
end

function dif_f_p = fx_second_derivative(fx,startingPoint,finishPoint,x)

    interval = finishPoint - startingPoint;
    h = interval * 10^-6;    
    dif_f_p = (fx(x + h) + fx(x - h) - 2*fx(x))/(h^2);
    
end

function [x] = Gradient_Descent(fx,startingPoint,finishPoint,iteration,etha)

    x0 = startingPoint + (finishPoint-startingPoint).*rand(1,1);
    for i = 1:iteration
        x0 = x0 - etha * fx_derivative(fx,startingPoint,finishPoint,x0);
    end
    x = x0;
    
end

function [x] = Newton_Rafson(fx,startingPoint,finishPoint,iteration)

    x0 = startingPoint + (finishPoint-startingPoint).*rand(1,1);
    for i = 1:iteration
        x = x0 - fx_derivative(fx,startingPoint,finishPoint,x0)...
            /fx_second_derivative(fx,startingPoint,finishPoint,x0);
    end
    x = x0;
    
end

function [x] = Simulated_Annual(fx,startPoint,finishPoint,iteration,gama,alpha)

t = 5;
x0 = startPoint + (finishPoint-startPoint).*rand(1,1);
x_prime = x0;
x = x_prime;
d = 0;
    
for i = 1:iteration
        
    if (fx(x0) < fx(x)) 
            x = x0;
    end
        
    Rx = -alpha + (alpha-(-alpha)).*rand(1,1);
        
    x1 = x0 + Rx;
        
    if (x1 > finishPoint)
        x1 = finishPoint;
    end
        
    if (x1 < startPoint)
        x1 = startPoint;
    end
        
    x_prime = x1;
        
    if (fx(x1) > fx(x0))
            
        if ((fx(x1) - fx(x0)) > d)
            d = fx(x1) - fx(x0);
        end
            
        Rx = rand(1,1);
            
        threshold = exp((fx(x0) - fx(x1))/(t*d));
            
        if (Rx > threshold)
            x_prime = x0;
        else
            x_prime = x1;
        end
    end
        
    x0 = x_prime;
    t = t * gama;
end
end




