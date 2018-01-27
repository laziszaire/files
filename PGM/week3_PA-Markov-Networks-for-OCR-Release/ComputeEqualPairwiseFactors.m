function factors = ComputeEqualPairwiseFactors (images, K)
% This function computes the pairwise factors for one word in which every
% factor value is set to be 1.
%
% Input:
%   images: An array of structs containing the 'img' value for each
%     character in the word.
%   K: The alphabet size (accessible in imageModel.K for the provided
%     imageModel).
%
% Output:
%   factors: The pairwise factors for this word. Every entry in the factor
%     vals should be 1.
%
% Copyright (C) Daphne Koller, Stanford University, 2012

n = length(images);

factors = repmat(struct('var', [], 'card', [], 'val', []), n - 1, 1);

% Your code here:
for i = 1:n-1
    factors(i).var = [i,i+1];
    factors(i).card = [K,K];
    index = 1:prod(factors(i).card);
    assignments = IndexToAssignment(index,factors(i).card);
    v = ones(size(index));
    factors(i) =  SetValueOfAssignment(factors(i), assignments, v);
end
