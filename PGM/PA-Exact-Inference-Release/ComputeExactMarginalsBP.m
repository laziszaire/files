%COMPUTEEXACTMARGINALSBP Runs exact inference and returns the marginals
%over all the variables (if isMax == 0) or the max-marginals (if isMax == 1).
%
%   M = COMPUTEEXACTMARGINALSBP(F, E, isMax) takes a list of factors F,
%   evidence E, and a flag isMax, runs exact inference and returns the
%   final marginals for the variables in the network. If isMax is 1, then
%   it runs exact MAP inference, otherwise exact inference (sum-prod).
%   It returns an array of size equal to the number of variables in the
%   network where M(i) represents the ith variable and M(i).val represents
%   the marginals of the ith variable.
%
% Copyright (C) Daphne Koller, Stanford University, 2012


function M = ComputeExactMarginalsBP(F, E, isMax)
if nargin<2,E=[];isMax=0;end
% initialization
% you should set it to the correct value in your code
M = [];
P = CreateCliqueTree(F, E);
P = CliqueTreeCalibrate(P, isMax);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%
% Implement Exact and MAP Inference.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-- given a calibrated tree how to compute marginals?
%-- a. marginalize any clique that contains it.
vars = unique([F.var]);
M = repmat(struct('var', 0, 'card', 0, 'val', []), length(vars), 1);
N = cellfun(@length,{P.cliqueList.var});
for i = 1:length(vars)
    v =vars(i);
    is_ = cellfun(@(x) ismember(v,x),{P.cliqueList.var});
    [~,idx]= min(N(is_));a = find(is_);
    clique_min = P.cliqueList(a(idx));
    switch isMax,
        case 0
            M(i) = FactorMarginalization(clique_min,setdiff(clique_min.var,v));
            M(i).val = M(i).val/sum(M(i).val);
        case 1
    end
end
end