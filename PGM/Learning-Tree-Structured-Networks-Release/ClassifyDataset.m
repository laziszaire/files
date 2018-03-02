function accuracy = ClassifyDataset(dataset, labels, P, G)
% returns the accuracy of the model P and graph G on the dataset 
%
% Inputs:
% dataset: N x 10 x 3, N test instances represented by 10 parts
% labels:  N x 2 true class labels for the instances.
%          labels(i,j)=1 if the ith instance belongs to class j 
% P: struct array model parameters (explained in PA description)
% G: graph structure and parameterization (explained in PA description) 
%
% Outputs:
% accuracy: fraction of correctly classified instances (scalar)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

N = size(dataset, 1);
accuracy = 0.0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
logCO = zeros(N,2);
for i = 1:N
    logCO(i,:) = logcO(squeeze(dataset(i,:,:)),P,G);
end
preds = logCO(:,1)>logCO(:,2);
accuracy = labels(:,1)==preds;
accuracy = sum(accuracy)/numel(accuracy);
fprintf('Accuracy: %.2f\n', accuracy);
end


function logCO = logcO(D1,P,G)
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
logCO = logO_cpd+logc;%factor product
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