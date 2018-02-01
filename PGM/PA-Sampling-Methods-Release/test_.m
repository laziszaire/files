%block sampling
load('exampleIOPA5.mat')
disp('logBS')
[V, G, F, A] = deal(exampleINPUT.t6a1,exampleINPUT.t6a2,exampleINPUT.t6a3,exampleINPUT.t6a4);
LogBS = BlockLogDistribution(V, G, F, A);
disp(~any(round(LogBS - exampleOUTPUT.t6)))


%gibbs trans
disp('gibbs trans')
clear
load('exampleIOPA5.mat')
[A, G, F] = deal(exampleINPUT.t7a1{1},exampleINPUT.t7a2{1},exampleINPUT.t7a3{1});
Anext = GibbsTrans(A, G, F);
disp(sum(Anext - exampleOUTPUT.t7{1})<.001)

%MCMC inference
[toy_network, toy_factors] = ConstructToyNetwork(1, .1);
nvar = length(toy_network.names);
A0 = exampleINPUT.t8a8{1};
E = exampleINPUT.t8a3{1};
[G,F,mix_time]= deal(exampleINPUT.t8a1{1},exampleINPUT.t8a2{1},exampleINPUT.t8a5{1});
[M, all_samples] = MCMCInference(G,F, E, 'Gibbs', mix_time, ...
                                                  exampleINPUT.t8a6{1}, 1, A0);
t8o1 = exampleOUTPUT.t8o1{1};

%MHUniformTrans    baseline
[Aout, G, F] = deal(exampleINPUT.t9a1{1},exampleINPUT.t9a2{1},exampleINPUT.t9a3{1});
all(exampleOUTPUT.t9{1} == Aout)

%MHSWTrans1

