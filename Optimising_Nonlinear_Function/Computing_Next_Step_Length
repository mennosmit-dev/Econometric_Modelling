function [lambda_next] = ComputeNextLambda(A, lambda, LB, UB, x_lagrange, rho)
% ComputeNextLambda calculates the next vector of Lagrangian multipliers
% using a subgradient step.
%
% Inputs:
%   A          - Adjacency matrix of the instance (m x n)
%   lambda     - Current vector of Lagrangian multipliers (m x 1)
%   LB         - Lower bound on the optimal primal objective value z*
%   UB         - Upper bound on the optimal primal objective value z*
%   x_lagrange - Optimal solution of the Lagrangian relaxation for lambda (n x 1)
%   rho        - Scaling factor for the subgradient step
%
% Output:
%   lambda_next - Updated vector of Lagrangian multipliers (m x 1)

m = size(A, 1);
ones_vector = ones(m, 1);

subgradient = ones_vector - A * x_lagrange;

if all(subgradient == 0)
    % If subgradient is zero vector, keep lambda unchanged
    lambda_next = lambda;
else
    mu = rho * (UB - LB) / (norm(subgradient)^2);
    lambda_next = lambda + mu * subgradient;
    
    % Ensure lambda_next is non-negative (project onto non-negative orthant)
    lambda_next(lambda_next < 0) = 0;
end

end
