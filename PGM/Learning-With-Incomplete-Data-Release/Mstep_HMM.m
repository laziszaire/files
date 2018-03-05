function P = Mstep_HMM(ClassProb,PairProb,G,actionData,poseData)
%MLE of parameters

K = size(ClassProb, 2);
% p(s) ==> P.c

s1_ind = cellfun(@(a) a(1),{actionData.marg_ind});
P.c = norm_(mean(ClassProb(s1_ind,:)));

% p(s'|s) ==> P.transMatrix

fh = @(pair_ind) mean(PairProb(pair_ind,:)); %todo fix logp ==>p
mean_action1 = cellfun(fh,{actionData.pair_ind},'uniformoutput',false);
mean_dataset = mean(cat(1,mean_action1{:}));
P.transMatrix = reshape(norm_(mean_dataset),K,[]);

%p(O|s) ==> P.clg
P.clg = est_clg(G,poseData,ClassProb);
end
%% subfunctions

function p = norm_(measure)
%normalize measure to get a probablity

p = measure/sum(measure);
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