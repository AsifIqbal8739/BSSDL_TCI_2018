function [dist ind] = dic_dist(D0,d0,D,d)

B = max(d); B0 = max(d0);
D = block_orthonormalize(D,d);
D0 = block_orthonormalize(D0,d0);
G = D'*D0;
if (B == size(D,2)) && (B0 == size(D0,2))
    dist0 = sqrt(1-G.^2);
else
    for i = 1:B
        di  = (d==i);
        s = sum(di);
        for j = 1:B0
            d0j  = (d0==j);
            s0 = sum(d0j);
            dist0(i,j) = sqrt(1-min([norm(G(di,d0j),'fro')^2/max([s s0]) 1]));
        end
    end
end

dist = zeros(1,B); ind = zeros(1,B);
for i = 1:B
    [mn r] = min(dist0); [mn c] = min(mn); r = r(c);
    if isinf(mn),   continue;   end             % My changes
    dist(r)=mn; ind(r)=c;
    dist0(:,c)=inf; dist0(r,:)=inf;
end

% Ind => columns correspond to rec block and value gives which orig block it is
% most similar to