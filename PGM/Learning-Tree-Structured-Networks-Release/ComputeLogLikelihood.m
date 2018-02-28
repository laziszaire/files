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
%marginal out C
for i = 1:N
    loglikelihood = loglikelihood+logPO_(squeeze(dataset(i,:,:)),P,G);
end
end


function logPO = logPO_(D1,P,G)
% G, graph structure,10 variable and their parent
% P, parameters, mu, sigma and theta,classp
% D, an instance, 10*3  numveriable*(y, x, alpha)

N_variables =size(D1,1);
cp = P.c;
logc = log(cp);
logO_cpd = 0*logc;
    for i = 1:N_variables
        logO_cpd = logO_cpd + logcpd(G(i,:),P.clg(i),D1(i,:),D1); %factor product
    end
logPO = log(sum(exp(logO_cpd+logc)));%marginal out C, get P(O)
end


function logO_cpd = logcpd(Gi,Pi_clg,Di,D1)
% 返回y，x，alpha的mu和sigma, 一列为一个分类class
% log(y)+log(x)+log(alpha)
sigmas = [Pi_clg.sigma_y;Pi_clg.sigma_x;Pi_clg.sigma_angle];
mus = zeros(3,2);
if Gi(1)==0
    %Gi只有C一个parent
    mus = [Pi_clg.mu_y;Pi_clg.mu_x;Pi_clg.mu_angle];
else
    parent = [1,D1(Gi(2),:)];
    theta_ = reshape(Pi_clg.theta',4,3,2);
    for i = 1:2
        mus(:,i) = parent*theta_(:,:,i);
    end
end
logO_cpd = logOi(Di,mus,sigmas); %log(O|c=k,Op)
end

function logO = logOi(Di,mus,sigmas)
% 计算Oi variable的条件log概率
% log(y)+log(x)+log(alpha)

logO = [0,0];
for j = 1:2
    for i = 1:3
        logO(j) = logO(j) + lognormpdf(Di(i),mus(i,j),sigmas(i,j));
    end
end
end