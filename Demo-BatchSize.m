%% Initialization
clear ; close all; clc

%% ======================= Plotting Data =======================
fprintf('Plotting Data ...\n')
data = load('ex1data1.txt');
X1 = data(:, 1); y = data(:, 2);
m = length(y); % number of training examples
% X1 = featureNormalize(X1);
% Plot Data
% Note: You have to complete the code in plotData.m
figure;
dataFigure1 = plotData(X1, y, 1);
dataFigure2 = plotData(X1, y, 2);

% ============= Visualizing J(theta_0, theta_1) =============
% X = data(:, 1);
% X = featureNormalize(X);
X1 = [ones(m, 1), X1]; % Add a column of ones to x
theta1 = [0; 0]; % initialize fitting parameters
theta2 = [0; 0]; % initialize fitting parameters

fprintf('Visualizing J(theta_0, theta_1) ...\n')

% Grid over which we will calculate J
theta0_vals = linspace(-10, 20, 100);
theta1_vals = linspace(-1, 6, 200);
% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));
% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];
	  J_vals(i,j) = computeCost(X1, y, t);
    end
end
% Because of the way meshgrids work in the surf command, we need to
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';
% Contour plot
costFigure1 = subplot(2,2,3);
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');

costFigure2 = subplot(2,2,4);
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');

pause(1);
% =================== Cost and Gradient descent ===================

% Some gradient descent settings
iterations = 1000;
alpha = 0.01;
batch_size = 10;

fprintf('\nRunning Gradient Descent ...\n')
% run gradient descent
% J_history = zeros(iterations, 1);
subplot(dataFigure1);
hold on
pData1 = plot(X1(:,2), X1*theta1, 'b-');
hold off
subplot(dataFigure2);
hold on
pData2 = plot(X1(:,2), X1*theta2, 'b-');
hold off

subplot(costFigure1);
hold on
pCost1 = plot(theta1(1), theta1(2), 'o', 'MarkerSize', 2, 'LineWidth', 2);
hold off
subplot(costFigure2);
hold on
pCost2 = plot(theta2(1), theta2(2), 'o', 'MarkerSize', 2, 'LineWidth', 2);
hold off
pause(3)

for iter = 1:iterations
    theta1 = gradientDescent(X1, y, theta1, alpha);
    
    num_batchiters = floor(m/batch_size);
    for batchiter = 1:num_batchiters
        llimit = (batchiter-1)*batch_size + 1;
        ulimit = (batchiter)*batch_size;
        theta2 = gradientDescent(X1(llimit:ulimit,:), y(llimit:ulimit), theta2, alpha);
        subplot(costFigure2);
        hold on
        pCost2 = plot(theta2(1), theta2(2), 'bo', 'MarkerSize', 2, 'LineWidth', 2);
        hold off
%         pause(1)
%         fprintf('%f %f %f\n', batchiter, llimit, ulimit);
    end
    llimit = (num_batchiters)*batch_size +1;
    if(llimit<m)
        theta2 = gradientDescent(X1(llimit:end,:), y(llimit:end), theta2, alpha);
    end
    
    subplot(dataFigure1);
    hold on
    delete(pData1)
    pData1 = plot(X1(:,2), X1*theta1, 'b-');
    hold off
    subplot(dataFigure2);
    hold on
    delete(pData2)
    pData2 = plot(X1(:,2), X1*theta2, 'b-');
    hold off
    
    subplot(costFigure1);
    hold on
    pCost1 = plot(theta1(1), theta1(2), 'bo', 'MarkerSize', 2, 'LineWidth', 2);
    hold off
    subplot(costFigure2);
    hold on
    pCost2 = plot(theta2(1), theta2(2), 'bo', 'MarkerSize', 2, 'LineWidth', 2);
    hold off
    
    pause(0.01)
end

% print theta to screen
fprintf('Theta found by gradient descent:\n');
fprintf('%f\n', theta1);
fprintf('%f\n', theta2);
