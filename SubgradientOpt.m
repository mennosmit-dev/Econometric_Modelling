function [ LB_best, UB_best, x_best, LB_list, UB_list ] = PerformSubgradientOptimization( c, A, lambda_init, rho_init, k )
%Functions as a lagrangian subgradient optimisation algorithm
%c: cost vector c of the instance,
%A: adjacency matrix A of the instance,
%lambda_init: initial vector of Lagrangian multipliers to be used,
%rho_init: initial scaling factor rho to be used in the subgradient step
%k: (stricly positive) number of subgradient iterations to perform,
%LB_best: best found lower bound on the optimal primal objective value z*,
%UB_best: best found upper bound on the optimal primal objective value z*,
%x_best: best found primal feasible solution,
%z* for each of the k iteritons,
%LB_list: vector of found lower bounds on the optimal primal objective
%value z* for each of the k iterations,
%UB_list: vector of found upper bound on the optimal primal objective value
%z* for each of the k iterations,
lambda = lambda_init;
LB_list = zeros(k,1);
UB_list = zeros(k,1);
noBestYet = true;

for h = 1:k
    [LB, x_lagrange] = CalculateLagrangian(c, A, lambda);
    [UB, x_feas] = ConvertInfeasToFeas(c, A, x_lagrange);
    LB_list(h) = LB;
    UB_list(h) = UB;
    if h == 1
        UB_best = UB;
    end
    lambda = ComputeNextLambda(A, lambda, LB, UB_best, x_lagrange, rho_init/k);
    if noBestYet == false
        if LB > LB_best
            LB_best = LB;
        end
        if UB < UB_best
            x_best = x_feas;
            UB_best = UB;
        end
    else % When there is no best yet
        LB_best = LB;
        x_best = x_feas;
        noBestYet = false;
    end
end
end
