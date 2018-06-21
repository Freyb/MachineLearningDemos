function [theta] = gradientDescent(X, y, theta, alpha)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
% ====================== YOUR CODE HERE ======================
% Instructions: Perform a single gradient step on the parameter vector
%               theta. 
%
% Hint: While debugging, it can be useful to print out the values
%       of the cost function (computeCost) and gradient here.
%
temp1 = theta(1)-alpha*sum((X*theta-y).*X(:,1))/m;
temp2 = theta(2)-alpha*sum((X*theta-y).*X(:,2))/m;
theta(1) = temp1;
theta(2) = temp2;
end
