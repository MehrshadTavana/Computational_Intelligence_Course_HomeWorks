function y = k_means(x, k, center_points)

n = size(x,2);
dist = zeros(n,k);

for temp = 1:2000
    for i=1:n 
        for j=1:k
            dist(i,j)=norm(x(:,i)-center_points(:,j),2);
        end
    end
    
    [~,index]=min(dist,[],2);
    
    for i=1:k 
        center_points(:,i) = mean(x(:,index'==i),2);
    end
    
    y = index;
end

end