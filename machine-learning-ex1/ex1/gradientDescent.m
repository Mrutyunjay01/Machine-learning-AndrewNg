function theta = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    %disp(theta) [0;0] 2*1 mat
    error = (X * theta) - y;
    grad = (alpha/m) * (X'*error);
    
    theta = theta - grad;
    %disp(size(theta));
    %disp(grad);
    %disp(J_history(iter));
    %disp(computeCost(X, y, theta))
    J_history(iter) = computeCost(X, y, theta);
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %







    % ============================================================

    % Save the cost J in every iteration    
end

end
