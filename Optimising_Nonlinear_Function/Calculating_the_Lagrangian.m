function [obj_lagrange, x_lagrange] = CalculateLagrangian(c, A, lambda)
% CalculateLagrangian computes the upper bound for the set covering problem
% by maximizing the Lagrangian relaxation.
%
% Inputs:
%   c      - Cost vector of the instance (n x 1)
%   A      - Adjacency matrix of the instance (m x n)
%   lambda - Vector of Lagrangian multipliers (m x 1)
%
% Outputs:
%   x_lagrange   - Optimal solution of the Lagrangian relaxation for lambda (n x 1)
%   obj_lagrange - Optimal objective value of the Lagrangian function for lambda (scalar)

n = length(c);
m = size(A, 1);

x_lagrange = zeros(n, 1);

% Compute optimal solution x_lagrange for given lambda
for j = 1:n
    if (c(j) - lambda' * A(:, j)) < 0
        x_lagrange(j) = 1;
    else
        x_lagrange(j) = 0;
    end
end

% Calculate the Lagrangian objective value
obj_lagrange = lambda' * ones(m, 1);

for j = 1:n
    obj_lagrange = obj_lagrange + (c(j) - lambda' * A(:, j)) * x_lagrange(j);
end

end

