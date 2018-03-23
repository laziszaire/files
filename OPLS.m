function X_orthogonal = OPLS(X,y,number_of_orthogonal_component)
%orthogonal projection onto latent structure

w = X'*y;
w = w/norm(w);
[M,N] = size(X) ;
T_orthogonal = zeros(M,number_of_orthogonal_component);
P_orthogonal = zeros(number_of_orthogonal_component,N)';
for i = 1:number_of_orthogonal_component
    t = X*w;
    p = X'*t/(t'*t);
    w_orthogonal = p - p'*w*w;
    w_orthogonal = w_orthogonal/norm(w_orthogonal);
    t_orthogonal = X*w_orthogonal;%X到w_orthogonal上的标量投影component
    p_orthogonal = X'*t_orthogonal/(t_orthogonal'*t_orthogonal);
    T_orthogonal(:,i) = t_orthogonal;
    P_orthogonal(:,i) = p_orthogonal;
end
X_orthogonal = T_orthogonal*P_orthogonal';

end
    