function [LB_best, UB_best, x_best, LB_list, UB_list] = PerformSubgradientOptimization(c, A, lambda_init, rho_init, k)
% PerformSubgradientOptimization runs the Lagrangian subgradient optimization algorithm.
%
% Inputs:
%   c           - Cost vector of the instance (n x 1)
%   A           - Adjacency matrix of the instance (m x n)
%   lambda_init - Initial vector of Lagrangian multipliers (m x 1)
%   rho_init    - Initial scaling factor for the subgradient step
%   k           - Number of subgradient iterations (strictly positive integer)
%
% Outputs:
%   LB_best     - Best found lower bound on the optimal primal objective value z*
%   UB_best     - Best found upper bound on the optimal primal objective value z*
%   x_best      - Best found feasible primal solution
%   LB_list     - Vector of lower bounds found at each iteration (k x 1)
%   UB_list     - Vector of upper bounds found at each iteration (k x 1)

lambda = lambda_init;
LB_list = zeros(k, 1);
UB_list = zeros(k, 1);

noBestYet = true;

for h = 1:k
    % Calculate Lagrangian lower bound and solution
    [LB, x_lagrange] = CalculateLagrangian(c, A, lambda);
    
    % Convert infeasible solution to feasible and calculate upper bound
    [UB, x_feas] = ConvertInfeasToFeas(c, A, x_lagrange);
    
    LB_list(h) = LB;
    UB_list(h) = UB;
    
    if h == 1
        UB_best = UB;
    end
    
    % Update Lagrangian multipliers using subgradient step
    lambda = ComputeNextLambda(A, lambda, LB, UB_best, x_lagrange, rho_init / k);
    
    if ~noBestYet
        % Update best lower bound if improved
        if LB > LB_best
            LB_best = LB;
        end
        
        % Update best feasible solution and upper bound if improved
        if UB < UB_best
            x_best = x_feas;
            UB_best = UB;
        end
    else
        % Initialize best values at first iteration
        LB_best = LB;
        x_best = x_feas;
        noBestYet = false;
    end
end

end
