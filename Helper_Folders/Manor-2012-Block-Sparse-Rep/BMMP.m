function [X nz_blocks nz_rw ] = BMMP(D, B, d, dd, thresh, max_it,s)

%Kevin Rosenblum - 2010
%BMMP executes the BOMP algorithm and finds s-block-sparse representations 
%of the signals B given a dictionary D with corresponding block structure d


p = 1; K = size(D,2);
O1 = ones(s,1);
run = (max(d) == K);
nz_blocks = []; nz_rw = []; S = [];
B0 = B; nB0 = norm(B0,'fro');

while p<=max_it &&(norm(B,'fro')/nB0>thresh)
    rr= [D'*B;0];
    if ~run
        r = rr(dd).^2*O1;
        [mx arg] = max(r);
    else
        [mx arg] = max(abs(rr));
    end
 
    nz_blocks = [nz_blocks arg];
    cols = find(d==arg);
    nz_rw = [nz_rw cols];
    s = D(:,cols);
    S = [S s];
    [Q,R] = qr(S,0);
    B = B-Q*(Q'*B);
    p = p+1;
end

X = zeros(K,1);
X(nz_rw) = (R)\(Q'*B0);
