% File: EM_HMM.m
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

function [P loglikelihood ClassProb PairProb] = EM_HMM(actionData, poseData, G, InitialClassProb, InitialPairProb, maxIter)

% INPUTS
% actionData: structure holding the actions as described in the PA
% poseData: N x 10 x 3 matrix, where N is number of poses in all actions
% G: graph parameterization as explained in PA description
% InitialClassProb: N x K matrix, initial allocation of the N poses to the K
%   states. InitialClassProb(i,j) is the probability that example i belongs
%   to state j.
%   This is described in more detail in the PA.
% InitialPairProb: V x K^2 matrix, where V is the total number of pose
%   transitions in all HMM action models, and K is the number of states.
%   This is described in more detail in the PA.
% maxIter: max number of iterations to run EM

% OUTPUTS
% P: structure holding the learned parameters as described in the PA
% loglikelihood: #(iterations run) x 1 vector of loglikelihoods stored for
%   each iteration
% ClassProb: N x K matrix of the conditional class probability of the N examples to the
%   K states in the final iteration. ClassProb(i,j) is the probability that
%   example i belongs to state j. This is described in more detail in the PA.
% PairProb: V x K^2 matrix, where V is the total number of pose transitions
%   in all HMM action models, and K is the number of states. This is
%   described in more detail in the PA.

% Initialize variables
N = size(poseData, 1);
K = size(InitialClassProb, 2);
L = size(actionData, 2); % number of actions
V = size(InitialPairProb, 1);

ClassProb = InitialClassProb; %class of pose p(S|P)
PairProb = InitialPairProb; % p(S|S')

loglikelihood = zeros(maxIter,1);

P.c = [];
P.clg.sigma_x = [];
P.clg.sigma_y = [];
P.clg.sigma_angle = [];
P.transMatrix = zeros(K,K);

% EM algorithm
for iter=1:maxIter
    
    % M-STEP to estimate parameters for Gaussians
    % Fill in P.c, the initial state prior probability (NOT the class probability as in PA8 and EM_cluster.m)
    % Fill in P.clg for each body part and each class
    % Make sure to choose the right parameterization based on G(i,1)
    % Hint: This part should be similar to your work from PA8 and EM_cluster.m
    
    %   P.c = zeros(1,K);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % YOUR CODE HERE
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % M-STEP to estimate parameters for transition matrix
    % Fill in P.transMatrix, the transition matrix for states
    % P.transMatrix(i,j) is the probability of transitioning from state i to state j
    %   P.transMatrix = zeros(K,K);
    
    % Add Dirichlet prior based on size of poseData to avoid 0 probabilities
    %   P.transMatrix = P.transMatrix + size(PairProb,1) * .05;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % YOUR CODE HERE
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    P = Mstep_HMM(ClassProb,PairProb,G,actionData,poseData);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    % E-STEP preparation: compute the emission model factors (emission probabilities) in log space for each
    % of the poses in all actions = log( P(Pose | State) )
    % Hint: This part should be similar to (but NOT the same as) your code in EM_cluster.m
    
    %   logEmissionProb = zeros(N,K);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % YOUR CODE HERE
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    % E-STEP to compute expected sufficient statistics
    % ClassProb contains the conditional class probabilities for each pose in all actions
    % PairProb contains the expected sufficient statistics for the transition CPDs (pairwise transition probabilities)
    % Also compute log likelihood of dataset for this iteration
    % You should do inference and compute everything in log space, only converting to probability space at the end
    % Hint: You should use the logsumexp() function here to do probability normalization in log space to avoid numerical issues
    
    ClassProb = zeros(N,K);
    PairProb = zeros(V,K^2);
    loglikelihood(iter) = 0;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % YOUR CODE HERE
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for i = 1:numel(actionData)
        [M, PCalibrated] = Estep1_HMM(P,G,actionData(i),poseData);
        ClassProb(actionData(i).marg_ind,:) = cat(1,M.val);
        %   MSS(i,:) = logsumexp(cat(1,{PCalibrated.val}));
        PairProb(actionData(i).pair_ind,:) = cat(1,PCalibrated.cliqueList.val);%logspace, cliqueList(22,23) =>(s,s')
    end
    ClassProb = exp(ClassProb);
    PairProb = exp(PairProb-logsumexp(PairProb));
    
    loglikelihood(iter) = logsumexp(PCalibrated.cliqueList(1).val);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Print out loglikelihood
    disp(sprintf('EM iteration %d: log likelihood: %f', ...
        iter, loglikelihood(iter)));
    if exist('OCTAVE_VERSION')
        fflush(stdout);
    end
    
    % Check for overfitting by decreasing loglikelihood
    if iter > 1
        if loglikelihood(iter) < loglikelihood(iter-1)
            break;
        end
    end
    
end

% Remove iterations if we exited early
loglikelihood = loglikelihood(1:iter);
end

function P = Mstep_HMM(ClassProb,PairProb,G,actionData,poseData)
%MLE of parameters

% ClassProb is the soft assignments of the hidden states in log space
K = size(ClassProb, 2);

% p(S1): K*1��K is number of class for pose

s1_ind = cellfun(@(a) a(1),{actionData.marg_ind});
P.c = norm_(sum(ClassProb(s1_ind,:)));

% p(S|S') 
% P.transMatrix
% pairProb -- p(s',s) == margianl out s' ==>p(s)
% transition  p(s'|s) = p(s',s)/p(s)

dirchlet = size(PairProb,1) * .05;
P.transMatrix = norm_m(sum(reshape(PairProb',K,K,[]),3)+dirchlet);%dirchlet

%p(O|s) ==> P.clg
P.clg = est_clg(G,poseData,ClassProb);
end

function clg = est_clg(G,dataset,classProb)
%
%dataset:posedata
clg.sigma_x = [];
clg.sigma_y = [];
clg.sigma_angle = [];
clg.mu_x = [];
clg.mu_y = [];
clg.mu_angle = [];
clg.theta = [];
K = size(classProb,2);
nO = size(G,1);
for k = 1:K
    Gk = G;if numel(size(G))==3,Gk = G(:,:,k);end
    W_k = classProb(:,k);
    for i_O = 1:nO
         X = squeeze(dataset(:,i_O,:));
        [y,x,angle] = deal(X(:,1),X(:,2),X(:,3));
        [has_parent,id_parent] = deal(Gk(i_O,1),Gk(i_O,2));
        if has_parent
            X_parent = squeeze(dataset(:,id_parent,:));
            [Beta(:,1),clg(i_O).sigma_y(1,k)]= FitLG(y,X_parent,W_k);
            [Beta(:,2),clg(i_O).sigma_x(1,k)]= FitLG(x,X_parent,W_k);
            [Beta(:,3),clg(i_O).sigma_angle(1,k)]= FitLG(angle,X_parent,W_k);
            theta = Beta([end,1:end-1],:);
            clg(i_O).theta(k,:) = theta(:);
        elseif ~has_parent
            [clg(i_O).mu_y(1,k),clg(i_O).sigma_y(1,k)]= FitG(y,W_k);
            [clg(i_O).mu_x(1,k),clg(i_O).sigma_x(1,k)]= FitG(x,W_k);
            [clg(i_O).mu_angle(1,k),clg(i_O).sigma_angle(1,k)]= FitG(angle,W_k);
        end
    end
end
end

function p = norm_m(measure)

p = bsxfun(@rdivide,measure,sum(measure,2));
end

function p = norm_(measure)

p = measure/sum(measure);
end