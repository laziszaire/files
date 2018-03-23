function [p_fit,res_norm,fh] = fit_(p0,x,y)
[p_fit,res_norm] = lsqcurvefit(@tofit,p0,x,y);
fh = @(x) tofit(p_fit,x);
end

function y = tofit(params,x)
%´ýÄâºÏº¯Êý
[A1,A2,x0,dx] = deal(params(1),params(2),params(3),params(4));
y = (A1-A2)./(1+exp((x-x0)/dx))+A2;
end