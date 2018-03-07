function [M, PCalibrated] = Estep_HMM(P,G,actionData1,poseData)
%soft assignement of hidden variables: marginal of hidden variables

K = numel(P.c);
F = get_factors(actionData1.marg_ind,actionData1.pair_ind,P,G,K,poseData);
[M, PCalibrated] = ComputeExactMarginalsHMM(F);

end

function fs = get_factors(marg_ind,pair_ind,P,G,K,poseData)
%factor value in log space

%p(S)
nS = numel(marg_ind);
pS = repmat(struct('var',[],'card',K,'val',log(P.c)),nS,1);
pS(1).val = log(P.c);
pS(1).var = 1;
fh = @(m) m/sum(m);
for i = 2:nS
    pS(i).var = i;
    pS(i).val = log(fh(exp(pS(i-1).val)*P.transMatrix));%logspace
end

%p(S'|S)
nSS = numel(pair_ind);
pSS = repmat(struct('var',[],'card',[K,K],'val',log(reshape(P.transMatrix,1,[]))),nSS,1);
for i = 1:nSS
    pSS(i).var = [i,i+1];
end

%p(P|S),observed P ==> phi(S)

pPS = repmat(struct('var',[],'card',K,'val',[]),nS,1);
for i = 1:nS
    logpPS = logpPS_(G,P,squeeze(poseData(i,:,:)),pS(i).val);
    pPS(i).val = logpPS;
    pPS(i).var = i;%ith hidden state of an action i.
end
fs = [pS;pSS;pPS];
end

function logpPS = logpPS_(G,P,pose,logS)
%p(P|S),observed P ==> phi(S)

nO = size(G,1);
logP = logS*0;
K = numel(logS);
for k = 1:K
    for i_O=1:nO
        sigmas = [P.clg(i_O).sigma_y(k),P.clg(i_O).sigma_x(k),P.clg(i_O).sigma_angle(k)];
        if isempty(P.clg(i_O).mu_y)
            %compute mu of CLG
            parent = [1,squeeze(pose(G(i_O,2),:))];
            theta_k = reshape(P.clg(i_O).theta(k,:),4,3);
            mus = parent*theta_k;
        else
            mus = [P.clg(i_O).mu_y(k),P.clg(i_O).mu_x(k),P.clg(i_O).mu_angle(k)];
        end
        logP(i_O) = logOi(pose,mus,sigmas);
    end
    logpPS = sum(logP)+logS;
end
end