function [ obj_lagrange, x_lagrange ] = CalculateLagrangian( c, A, lambda )
%This function creates an upperbound for the set covering problem by
%maximising the lagrange relexation
%c: cost vector c of the instance,
%A: Adjacency matrix A of the instance,
%lambda: vector of Lagrangian multipliers,
%x_lagrange: optimal solution of the lagrange relaxation for lambda
%obj_lagrange: optimal objective value of the lagrangian function for
%lambda
n = length(c);
m = length(A(:,1));
x_lagrange = zeros(n,1);
for j = 1:n % For loop that calculates lagrange optimal solution
    if (c(j) - lambda'*A(:,j)) < 0
        x_lagrange(j) =  1;
    else
        x_lagrange(j) = 0;
    end
end
obj_lagrange = lambda'*ones(m,1); %first part of lagrange function
for j = 1:n % For loop that calculates lagrange optimal objective value
    obj_lagrange = obj_lagrange + (c(j) - lambda'*A(:,j))*x_lagrange(j);
end
end
