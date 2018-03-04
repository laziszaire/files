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