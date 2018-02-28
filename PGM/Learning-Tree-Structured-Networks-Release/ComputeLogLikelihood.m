function loglikelihood = ComputeLogLikelihood(P, G, dataset)
% returns the (natural) log-likelihood of data given the model and graph structure
%
% Inputs:
% P: struct array parameters (explained in PA description)
% G: graph structure and parameterization (explained in PA description)
%
%    NOTICE that G could be either 10x2 (same graph shared by all classes)
%    or 10x2x2 (each class has its own graph). your code should compute
%    the log-likelihood using the right graph.
%
% dataset: N x 10 x 3, N poses represented by 10 parts in (y, x, alpha)
% 
% Output:
% loglikelihood: log-likelihood of the data (scalar)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

N = size(dataset,1); % number of examples
K = length(P.c); % number of classes

loglikelihood = 0;
% You should compute the log likelihood of data as in eq. (12) and (13)
% in the PA description
% Hint: Use lognormpdf instead of log(normpdf) to prevent underflow.
%       You may use log(sum(exp(logProb))) to do addition in the original
%       space, sum(Prob).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end

function logp = logP(O)
%loglikelihood of a instance

end

function [mus, sigmas] = conditionalP(Oi,P,Opi,Gj)
%mus, sigmas for y, x,
parent = [1,Gj(Opi,:)];
theta = P.clg(Oi).theta;
theta_ = reshape(theta',4,3,2);
k = size(theta_,3);
mus = zeros(3,k);
if isempty(theta)
    %除了class label 没有其他parent
    sigmas = [P.clg(Oi).sigma_y;P.clg(Oi).sigma_x;P.clg(Oi).sigma_angle;];
    mus = [P.clg(Oi).mu_y;P.clg(Oi).mu_x;P.clg(Oi).mu_angle;];
else
   sigmas = [P.clg(2).sigma_y;P.clg(2).sigma_x;P.clg(2).sigma_angle;];
   for i = 1:k
       mus(:,i) = parent*(theta_(:,:,i));
   end
end
end