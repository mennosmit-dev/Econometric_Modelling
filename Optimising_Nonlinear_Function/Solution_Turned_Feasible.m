function [obj_feas, x_feas] = ConvertInfeasToFeas(c, A, x_infeas)
% ConvertInfeasToFeas constructs a feasible solution from an infeasible
% primal solution by applying a greedy heuristic for the set covering problem.
%
% Inputs:
%   c       - Cost vector of the instance (n x 1)
%   A       - Adjacency matrix of the instance (m x n)
%   x_infeas - Potentially infeasible primal solution (n x 1)
%
% Outputs:
%   obj_feas - Objective value of the feasible solution
%   x_feas   - Feasible primal solution constructed from x_infeas

n = length(c);
m = size(A, 1);

x_feas = x_infeas;  % Start with the given solution
customersSatisfied = false;  % Flag indicating whether all constraints are satisfied

while ~customersSatisfied
    unsatisfiedCount = 0;  % Count how many constraints are not satisfied
    
    % Check constraints satisfaction
    for i = 1:m
        coverage = sum(A(i, :) .* x_feas');
        if coverage == 0
            unsatisfiedCount = unsatisfiedCount + 1;
        end
    end
    
    if unsatisfiedCount == 0
        customersSatisfied = true;
    else
        % Add the best route to cover unsatisfied constraints
        bestIndex = 0;
        bestRatio = -inf;
        
        for j = 1:n
            if x_feas(j) == 0  % Consider only routes not already selected
                % Calculate added coverage for uncovered customers by adding route j
                addedCoverage = 0;
                for i = 1:m
                    % Check if customer i is already covered
                    isCovered = any(x_feas' .* A(i, :));
                    if ~isCovered && A(i, j) == 1
                        addedCoverage = addedCoverage + 1;
                    end
                end
                
                % If cost is zero, immediately select this route
                if c(j) == 0
                    bestIndex = j;
                    break;
                else
                    ratio = addedCoverage / c(j);
                    if ratio > bestRatio
                        bestRatio = ratio;
                        bestIndex = j;
                    end
                end
            end
        end
        
        x_feas(bestIndex) = 1;  % Add the best route found
    end
end

obj_feas = c' * x_feas;  % Calculate the objective value of the feasible solution

end
