%%

%%
load PA8SampleCases.mat
P = exampleINPUT.t3a1;
G = exampleINPUT.t3a2;
dataset = exampleINPUT.t3a3;
 D1 = squeeze(dataset(1,:,:));
 loglikelihood = ComputeLogLikelihood(P, G, dataset);
 loglikelihood-exampleOUTPUT.t3

 
 %% learning parameters with known graph structures
 % 可能有多个图结构
 clear
load PA8SampleCases.mat
dataset = exampleINPUT.t4a1;
G = exampleINPUT.t4a2;
labels = exampleINPUT.t4a3;
[P,loglikelihood] = LearnCPDsGivenGraph(dataset, G, labels);
% P没问题
% loglikelihood 也怎么问题