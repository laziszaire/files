transition_names = 'Gibbs';
ExactM = ComputeExactMarginalsBP(toy_factors, toy_evidence, 0);
figure, VisualizeToyImageMarginals(toy_network, ExactM,1,'exact');
rand('seed', 1);
for i = 1:2
A0 = i * ones(1, length(toy_network.names));
[M, all_samples] = ...
    MCMCInference(toy_network, toy_factors, toy_evidence, transition_names, 0, 4000, 1, A0);
figure, VisualizeToyImageMarginals(toy_network, M, i, transition_names);
end