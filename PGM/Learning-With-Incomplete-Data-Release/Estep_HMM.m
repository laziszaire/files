function logEmissionProb = Estep_HMM(P,G,actionData,poseData)
%
ComputeExactMarginalsHMM(F)

end

function fs = get_factors(marg_ind,P,pair_ind,K,poseData)

%p(S)
nS = numel(marg_ind);
pS = repmat(struct('var',[],'card',K,'val',P.c),nS,1);
for i = 1:nS
pS(i).var = marg_ind;
end

%p(S'|S)
nSS = numel(pair_ind);
pS = repmat(struct('var',[],'card',9,'val',reshape(P.transMatrix,1,[])),nSS,1);
for i = 1:nSS
    pS(i).var = [pair_ind(i),pair_ind(i)+1];
end

%p(P|S),observed P ==> phi(S)

pPS = repmat(struct('var',[],'card',K,'val',[]),nS,1);
for i = 1:nS
logpPS = logpPS_(G,P,squeeze(poseData(i,:,:)),P.c);
pPS(i).val = logpPS;
end 
end

function logpPS = logpPS_(G,P,pose,logS)
%p(P|S),observed P ==> phi(S)

nO = size(G,1);
logP = logS*0;
K = numel(logS);
for k = 1:K
for i_O=1:nO
    sigmas = [P(i_O).sigma_y(k),P(i_O).sigma_x(k),P(i_O).sigma_angle(k)];
    if isempty(P(i_O).mu_y)
        %compute mu of CLG
        parent = [1,squeeze(pose(G(i_O,2),:))];
        theta_k = reshape(P(i_O).theta(k,:),4,3);
        mus = parent*theta_k;
    else 
        mus = [P(i_O).mu_y(k),P(i_O).mu_x(k),P(i_O).mu_angle(k)];
    end
    logP(i_O) = logOi(pose,mus,sigmas);
end
logpPS = sum(logP)+logS;
end
end