clear
load PA9SampleCases
[poseData, G, InitialClassProb, maxIter] = deal(exampleINPUT.t1a1,...
                                                exampleINPUT.t1a2,...
                                                exampleINPUT.t1a3,...
                                                exampleINPUT.t1a4);
[P,loglikelihood,ClassProb] = EM_cluster(poseData, G, InitialClassProb, maxIter);

%%
clear
load PA9Data.mat
[P,loglikelihood,ClassProb] = EM_cluster(poseData1, G, InitialClassProb1, 20);
%% EM_HMM
clear
load PA9SampleCases
[actionData, poseData, G, InitialClassProb, InitialPairProb, maxIter] = deal(exampleINPUT.t2a1,...
                                                exampleINPUT.t2a2,...
                                                exampleINPUT.t2a3,...
                                                exampleINPUT.t2a4,...
                                                exampleINPUT.t2a5,...
                                                exampleINPUT.t2a6);
[P,loglikelihood,ClassProb,PairProb] = EM_HMM(actionData, poseData, G, InitialClassProb, InitialPairProb, maxIter);
% %P.transMatrix check
% ClassProb check
% PairProb check

% [P loglikelihood ClassProb PairProb] = EM_HMM_(actionData, poseData, G, InitialClassProb, InitialPairProb, maxIter)