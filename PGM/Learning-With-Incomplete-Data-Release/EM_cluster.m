% File: EM_cluster.m
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

function [P,loglikelihood,ClassProb] = EM_cluster(poseData, G, InitialClassProb, maxIter)

% INPUTS
% poseData: N x 10 x 3 matrix, where N is number of poses;
%   poseData(i,:,:) yields the 10x3 matrix for pose i.
% G: graph parameterization as explained in PA8
% InitialClassProb: N x K, initial allocation of the N poses to the K
%   classes. InitialClassProb(i,j) is the probability that example i belongs
%   to class j
% maxIter: max number of iterations to run EM

% OUTPUTS
% P: structure holding the learned parameters as described in the PA
% loglikelihood: #(iterations run) x 1 vector of loglikelihoods stored for
%   each iteration
% ClassProb: N x K, conditional class probability of the N examples to the
%   K classes in the final iteration. ClassProb(i,j) is the probability that
%   example i belongs to class j

% Initialize variables
N = size(poseData, 1);
K = size(InitialClassProb, 2);
ClassProb = InitialClassProb;
loglikelihood = zeros(maxIter,1);

P.c = [];
P.clg.sigma_x = [];
P.clg.sigma_y = [];
P.clg.sigma_angle = [];
P.clg.mu_x = [];
P.clg.mu_y = [];
P.clg.mu_angle = [];
P.clg.theta = [];
% EM algorithm
for iter=1:maxIter
    
    % M-STEP to estimate parameters for Gaussians
    %
    % Fill in P.c with the estimates for prior class probabilities
    % Fill in P.clg for each body part and each class
    % Make sure to choose the right parameterization based on G(i,1)
    %
    % Hint: This part should be similar to your work from PA8
    
%     P.c = zeros(1,K);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % YOUR CODE HERE
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    P = Mstep(ClassProb,G,poseData);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % E-STEP to re-estimate ClassProb using the new parameters
    %
    % Update ClassProb with the new conditional class probabilities.
    % Recall that ClassProb(i,j) is the probability that example i belongs to
    % class j.
    %
    % You should compute everything in log space, and only convert to
    % probability space at the end.
    %
    % Tip: To make things faster, try to reduce the number of calls to
    % lognormpdf, and inline the function (i.e., copy the lognormpdf code
    % into this file)
    %
    % Hint: You should use the logsumexp() function here to do
    % probability normalization in log space to avoid numerical issues
    
%     ClassProb = zeros(N,K);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % YOUR CODE HERE
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ClassProb = E_step(P,G,poseData);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Compute log likelihood of dataset for this iteration
    % Hint: You should use the logsumexp() function here
%     loglikelihood(iter) = 0;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % YOUR CODE HERE
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     loglikelihood(iter) = ComputeLogLikelihood(P, G, poseData);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Print out loglikelihood
    disp(sprintf('EM iteration %d: log likelihood: %f', ...
        iter, loglikelihood(iter)));
    if exist('OCTAVE_VERSION')
        fflush(stdout);
    end
    
    % Check for overfitting: when loglikelihood decreases
    if iter > 1
        if loglikelihood(iter) < loglikelihood(iter-1)
            break;
        end
    end
    
end

% Remove iterations if we exited early
loglikelihood = loglikelihood(1:iter);
end

function P = Mstep(ClassProb,G,dataset)
%
P.c = mean(ClassProb);
% ClassProb 是每个instance对class 影响的权重
K = size(ClassProb,2);
[~,nO,nvs] = size(dataset);%vs= {'y','x','alpha'};
Beta = zeros((nvs+1),nvs);
for k = 1:K
    W_k = ClassProb(:,k);%influence weights of instance for class k
    Gk=G;if numel(size(G))==3,Gk = squeeze(G(:,:,k));end
    for i_O = 1:nO
        [Oi_has_parent,Oi_parent_id] = deal(Gk(i_O,1),Gk(i_O,1));
        X = squeeze(dataset(:,i_O,:));
        [y,x,angle] = deal(X(:,1),X(:,2),X(:,3));
        if ~Oi_has_parent
            [P.clg(i_O).mu_y(1,k),P.clg(i_O).sigma_y(1,k)]= FitG(y,W_k);
            [P.clg(i_O).mu_x(1,k),P.clg(i_O).sigma_x(1,k)]= FitG(x,W_k);
            [P.clg(i_O).mu_angle(1,k),P.clg(i_O).sigma_angle(1,k)]= FitG(angle,W_k);
        elseif Oi_has_parent
            % parameters for P(Oi|C,Opi) = P(yi|c,Opi)*P(xi|c,Opi)*P(anglei|c,Opi)
            X_parent = squeeze(dataset(:,Oi_parent_id,:));
            %cellfun(@(e) all(e=='y'),vs)
            [Beta(:,1),P.clg(i_O).sigma_y(1,k)]= FitLG(y,X_parent,W_k);
            [Beta(:,2),P.clg(i_O).sigma_x(1,k)]= FitLG(x,X_parent,W_k);
            [Beta(:,3),P.clg(i_O).sigma_angle(1,k)]= FitLG(angle,X_parent,W_k);
            theta = Beta([end,1:end-1],:);
            P.clg(i_O).theta(k,:) = theta(:);
        end
    end
end
end

function ClassProb = E_step(P,G,dataset)
%conditional class prob: P(c|O)

[M,~,~] =size(dataset);
K = numel(P.clg(1).sigma_y);
logCO = zeros(M,K);
% logPO = logCO;
for i = 1:M
    %joint
    logCO(i,:) = logcO(squeeze(dataset(i,:,:)),P,G);
%     logPO(i,:) = logPO_(squeeze(dataset(i,:,:)),P,G);
end
% ClassProb = exp(logCO - logPO);
% normalize
logCO_cpd = bsxfun(@minus,logCO,logsumexp(logCO));
ClassProb = exp(logCO_cpd);
end

function loglikelihood = ComputeLogLikelihood(P, G, dataset)
%logp(O)
N_instance = size(dataset,1);
loglikelihood =0;
for i = 1:N_instance
    loglikelihood = loglikelihood+logPO_(squeeze(dataset(i,:,:)),P,G);
end
end

function logPO = logPO_(Os,P,G)
% G, graph structure,10 variable,their parent and class
% P, parameters, mu, sigma and theta,classp
% D, an instance, 10*3  numveriable*(y, x, alpha)

N_variables =size(Os,1);
cp = P.c;
logc = log(cp);
logO_cpd = 0*logc;
for i = 1:N_variables
    logO_cpd(i,:) = logcpd(squeeze(G(i,:,:)),P.clg(i),Os(i,:),Os); %(y,x,alpha) product
end
% prodcut of [P(O1|c),P(O2|c),...]
% bayesian P(O,c) = P(O|c)p(c)
logP = logc+sum(logO_cpd);
%marginal out c: P(O) = for i in numel(c),sum(P(O,c(i)))
logPO = logsumexp(logP); %%
end

function logCO = logcO(D1,P,G)
% joint p(c,O)
% G, graph structure,10 variable and their parent
% P, parameters, mu, sigma and theta,classp
% D, an instance, 10*3  numveriable*(y, x, alpha)

%return joint P(C,O)
N_variables =size(D1,1);
cp = P.c;
logc = log(cp);
logO_cpd = 0*logc;
    for i = 1:N_variables
        logO_cpd = logO_cpd + logcpd(G(i,:),P.clg(i),D1(i,:),D1); %factor product
    end
logCO = logO_cpd+logc; %bayessian rule
end


function logO_cpd = logcpd(Gi,Pi_clg,Oi,Os)
% log(y)+log(x)+log(alpha)
%Gi为2*numberclass

%logO_cpd :[1,K]
Gi = reshape(Gi,2,[]);
K = numel(Pi_clg.sigma_y);
if size(Gi,2)==1,Gi = repmat(Gi,1,K);end
logO_cpd = zeros(1,K);
for k = 1:K
    sigmas_k = [Pi_clg.sigma_y(k);Pi_clg.sigma_x(k);Pi_clg.sigma_angle(k)];
    if Gi(1,k)==0  %Gaussian
        mus_k= [Pi_clg.mu_y(k);Pi_clg.mu_x(k);Pi_clg.mu_angle(k)];
    else           %CLG
        theta_ = reshape(Pi_clg.theta',4,3,K);%[1,y,x,alpha]*[y,x,alpha]*num_class
        parent = [1,Os(Gi(2,k),:)];
        mus_k = parent*squeeze(theta_(:,:,k));
    end
    logO_cpd(k) = logOi(Oi,mus_k,sigmas_k); %log(O|c=k,Op)
end

end

function logO = logOi(Di,mus,sigmas)
% 计算Oi variable的条件log概率
% log(y)+log(x)+log(alpha)

num_yxalpha = numel(mus);
logO = 0;
for i = 1:num_yxalpha
    logO = logO+ lognormpdf(Di(i),mus(i),sigmas(i));
end
end