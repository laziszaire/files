% function [nll, grad] = InstanceNegLogLikelihood(X, y, theta, modelParams)
% returns the negative log-likelihood and its gradient, given a CRF with parameters theta,
% on data (X, y).
%
% Inputs:
% X            Data.                           (numCharacters x numImageFeatures matrix)
%              X(:,1) is all ones, i.e., it encodes the intercept/bias term.
% y            Data labels.                    (numCharacters x 1 vector)
% theta        CRF weights/parameters.         (numParams x 1 vector)
%              These are shared among the various singleton / pairwise features.
% modelParams  Struct with three fields:
%   .numHiddenStates     in our case, set to 26 (26 possible characters)
%   .numObservedStates   in our case, set to 2  (each pixel is either on or off)
%   .lambda              the regularization parameter lambda
%
% Outputs:
% nll          Negative log-likelihood of the data.    (scalar)
% grad         Gradient of nll with respect to theta   (numParams x 1 vector)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

function [nll, grad] = InstanceNegLogLikelihood(X, y, theta, modelParams)

% featureSet is a struct with two fields:
%    .numParams - the number of parameters in the CRF (this is not numImageFeatures
%                 nor numFeatures, because of parameter sharing)
%    .features  - an array comprising the features in the CRF.
%
% Each feature is a binary indicator variable, represented by a struct
% with three fields:
%    .var          - a vector containing the variables in the scope of this feature
%    .assignment   - the assignment that this indicator variable corresponds to
%    .paramIdx     - the index in theta that this feature corresponds to
%
% For example, if we have:
%
%   feature = struct('var', [2 3], 'assignment', [5 6], 'paramIdx', 8);
%
% then feature is an indicator function over X_2 and X_3, which takes on a value of 1
% if X_2 = 5 and X_3 = 6 (which would be 'e' and 'f'), and 0 otherwise.
% Its contribution to the log-likelihood would be theta(8) if it's 1, and 0 otherwise.
%
% If you're interested in the implementation details of CRFs,
% feel free to read through GenerateAllFeatures.m and the functions it calls!
% For the purposes of this assignment, though, you don't
% have to understand how this code works. (It's complicated.)

featureSet = GenerateAllFeatures(X, modelParams);
% X is not variable

% Use the featureSet to calculate nll and grad.
% This is the main part of the assignment, and it is very tricky - be careful!
% You might want to code up your own numerical gradient checker to make sure
% your answers are correct.
%
% Hint: you can use CliqueTreeCalibrate to calculate logZ effectively.
%       We have halfway-modified CliqueTreeCalibrate; complete our implementation
%       if you want to use it to compute logZ.

nll = 0;
grad = zeros(size(theta));

%%%
% Your code here:
% build clique tree
features = featureSet.features;
factors = features2factors(features,modelParams,theta);
CliqueTree = CreateCliqueTree(factors); % F is array of factors, it is only to be a chain.
[CliqueTree, logZ] = CliqueTreeCalibrate(CliqueTree,0); 

%nll
logz = logZ;
% The weighted feature counts
nnl_f = weighted_feature_counts(features,theta);
nll_reg = lambda*sum(theta.^2)/2;
nll = logz - nnl_f + nll_reg;


% grad
modelfc = 0*theta;
datafc = 0*theta;
grad_reg = lambda*theta;
grad = modelfc - datafc + grad_reg;

end
%% subfuncs
function factors = features2factors(features,modelParams,theta)
%make factors from features

Nf = numel(features);
factors = repmat(EmptyFactorStruct(),Nf,1);% one feature, one factor

for i = 1:Nf
    factors(i).var = features(i).var;
    factors(i).card = ones(1,numel(features(i).var))*modelParams.numHiddenStates;
    a = zeros(prod(factors(i).card));
    factors(i).val = exp(a);
    val_indicator_assingment = exp(theta(features(i).paramIdx)*1);
    factors(i) = SetValueOfAssignment(factors(i),features(i).assignment,val_indicator_assingment);
end

end