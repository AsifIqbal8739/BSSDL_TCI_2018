function [D,X,EE] = BlockKSVD(Y,D,X,Catoms,d_ini,B,verbb)
if ~exist('verbb','var'); verbb = 0; end
E = Y - D*X;
s = size(D,2)/B;
EE = 0;
for i = 1:B
    ii = d_ini == i;  
    if size(Catoms,1) == 1; indd = (i == Catoms);   
    else;    indd = any(i == Catoms);   end
    Ei = E(:,indd) + D(:,ii)*X(ii,indd);
    if nnz(indd) <= s
        [d,ss,vv] = svds(E(:,randperm(size(Y,2),B)),s);
        D(:,ii) = d;
%         X(ii,indd) = ss*vv';
%         fprintf('I''m Here\n');
        continue;
    end
    [U,S,V] = svds(Ei,s);
    D(:,ii) = U;
    X(ii,indd) = S*V';
    E(:,indd) = Ei - D(:,ii)*X(ii,indd);
end
if verbb
    EE = DispError(Y,D,X,verbb);
end
end