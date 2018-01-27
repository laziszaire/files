%CLIQUETREECALIBRATE Performs sum-product or max-product algorithm for
%clique tree calibration.

%   P = CLIQUETREECALIBRATE(P, isMax) calibrates a given clique tree, P
%   according to the value of isMax flag. If isMax is 1, it uses max-sum
%   message passing, otherwise uses sum-product. This function
%   returns the clique tree where the .val for each clique in .cliqueList
%   is set to the final calibrated potentials.
%
% Copyright (C) Daphne Koller, Stanford University, 2012

function P = CliqueTreeCalibrate(P, isMax)


% Number of cliques in the tree.
N = length(P.cliqueList);

% Setting up the messages that will be passed.
% MESSAGES(i,j) represents the message going from clique i to clique j.
MESSAGES = repmat(struct('var', [], 'card', [], 'val', []), N, N);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% We have split the coding part for this function in two chunks with
% specific comments. This will make implementation much easier.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% YOUR CODE HERE
% While there are ready cliques to pass messages between, keep passing
% messages. Use GetNextCliques to find cliques to pass messages between.
% Once you have clique i that is ready to send message to clique
% j, compute the message and put it in MESSAGES(i,j).
% Remember that you only need an upward pass and a downward pass.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[i,j] = GetNextCliques(P,MESSAGES);
a = [];
while i ~=0,
    %--- compute the message, marginalize the shared variables
    % variable elimination
    receive = setdiff(find(P.edges(:,i)>0),j);
    prod = FactorsProduct([P.cliqueList(i);MESSAGES(receive,i)]);%product
    message_var = intersect(prod.var,P.cliqueList(j).var);
    MESSAGES(i,j) = FactorMarginalization(prod, setdiff(prod.var,message_var));%sum
    MESSAGES(i,j) = normalize(MESSAGES(i,j));
    %--- pass the message, factor product, joint
    
    %P.cliqueList(j) = FactorProduct(P.cliqueList(j),MESSAGES(i,j));
%     if ~isempty(MESSAGES(j,i).val)
%         error = sum(abs(MESSAGES(i,j).val-MESSAGES(j,i).val))
%         disp(MESSAGES(i,j))
%         disp(MESSAGES(j,i))
%         MESSAGES(j,i) = struct('var', [], 'card', [], 'val', []);
%     end
    [i,j] = GetNextCliques(P,MESSAGES);
    a = [a;[i,j]];
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%
% Now the clique tree has been calibrated.
% Compute the final potentials for the cliques and place them in P.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i= 1:size(MESSAGES,1)
    P.cliqueList(i) = FactorsProduct([P.cliqueList(i); MESSAGES(P.edges(:,i)>0,i)]);%product
end
end

function message = normalize(message)
%
message.val = message.val/sum(message.val);
end