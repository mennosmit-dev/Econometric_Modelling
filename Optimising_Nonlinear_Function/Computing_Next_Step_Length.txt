
function [lambda_next] = ComputeNextLambda( A, lambda, LB, UB, x_lagrange, rho )
%Calculates the next lambda that needs to be used
%A: adjacency matrix A of the instance,
%lambda: vector of Lagrangian multipliers,
%LB: lower bound on the optimal primal objective value z*,
%UB: upper bound on the optimal primal objective value z*,
%x_lagrange: optimal solution of the Lagrangian relaxation for lambda,
%rho: sclaing factor to be used in the subgradient step, refer to pages

%lambda_next: new vector of lagrangian multipliers, constructed by applying
%the subgradient step

m = length(A(:,1));
vectorWithOnes = ones(m,1);
if (vectorWithOnes - A*x_lagrange) == zeros(m,1)
    lambda_next = lambda;
else
    mu = rho*(UB - LB)/norm(vectorWithOnes - A*x_lagrange)^2;
    lambda_next = lambda + mu*(vectorWithOnes - A*x_lagrange);
    lambdaBiggerThanZero = (lambda_next > 0);

    for i = 1:m
        if lambdaBiggerThanZero(i) == 0
            lambda_next(i) = 0;
        end
    end
end
end
