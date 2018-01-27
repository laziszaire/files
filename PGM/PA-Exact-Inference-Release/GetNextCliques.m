%GETNEXTCLIQUES Find a pair of cliques ready for message passing
%   [i, j] = GETNEXTCLIQUES(P, messages) finds ready cliques in a given
%   clique tree, P, and a matrix of current messages. Returns indices i and j
%   such that clique i is ready to transmit a message to clique j.
%
%   We are doing clique tree message passing, so
%   do not return (i,j) if clique i has already passed a message to clique j.
%
%	 messages is a n x n matrix of passed messages, where messages(i,j)
% 	 represents the message going from clique i to clique j.
%   This matrix is initialized in CliqueTreeCalibrate as such:
%      MESSAGES = repmat(struct('var', [], 'card', [], 'val', []), N, N);
%
%   If more than one message is ready to be transmitted, return
%   the pair (i,j) that is numerically smallest. If you use an outer
%   for loop over i and an inner for loop over j, breaking when you find a
%   ready pair of cliques, you will get the right answer.
%
%   If no such cliques exist, returns i = j = 0.
%
%   See also CLIQUETREECALIBRATE
%
% Copyright (C) Daphne Koller, Stanford University, 2012


function [i, j] = GetNextCliques(P, messages)

% initialization
% you should set them to the correct values in your code
i = 0;
j = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% messages that have passed
res =[];
n = size(messages,1);
for i = 1:n
    for j = 1:n
        %--- j is neighbor of i
        is_neighbor = P.edges(i,j)>0;
        if ~is_neighbor, continue,end
        %--- j has not received message from i
        jnot_receivedi = isempty(messages(i,j).var);
        if ~jnot_receivedi, continue,end
        %--- i received all the neighbors's message expect j
        neighbors = find(P.edges(:,i)>0);% all neighbors of i
        neighbors_except_j = setdiff(neighbors,j);
        if isempty(neighbors_except_j),%i has no neighbor except j, leaf
            ireceived = true;
        else
            not_empty = @(a) ~isempty(a);
            ireceived = all(cellfun(not_empty,{messages(neighbors_except_j,i).var}));
        end
        if ~ireceived, continue,end
        if all([is_neighbor,jnot_receivedi,ireceived])
            res = [i,j];
            break % find it and break
        end
    end
    if ~isempty(res),break,end
end
if isempty(res),i=0;j=0;return,end
[i,j] = deal(res(1),res(2));

end
