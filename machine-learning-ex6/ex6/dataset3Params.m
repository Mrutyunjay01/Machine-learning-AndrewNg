function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
Values = [0.01, 0.03, 0.1, 1, 3, 10, 30];
results = [];

for i=1:length(Values)
    for j =1:length(Values)
        
        C_i = Values(i);
        S_j = Values(j);
        
        model = svmTrain(X, y, C_i, @(x1,x2) gaussianKernel(x1, x2, S_j));
        predictions = svmPredict(model, Xval);
        
        error = mean(double(predictions ~= yval));
        
        %fprintf("C: %f\nsigma: %f\nerror: %f\n", C_i, S_j, error);
        
        results = [results; C_i, S_j, error];
    end
end

%disp(results);
% Now we need to calculate C and sigma associated with min error
%disp(min(results(:, 3)));
[~, MinIndex] = min(results(:, 3));
C = results(MinIndex, 1);
sigma = results(MinIndex, 2);
% extract from the result array





% =========================================================================

end
