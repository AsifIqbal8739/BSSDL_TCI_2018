
function [D1,D2,d2,D3] = compare_ksvd(Y,K,k,s,d0,max_it);
% [D1,D2,d2,D3] = compare_ksvd(Y,K,k,s,d0,max_it);
% train dictionaries using KSVD and BKSVD+SAC
%
% Output: 
%  D1 - result of KSVD
%  D2 - result of BKSVD+SAC
%  d2 - recovered block structure for BKSVD+SAC
%  D3 - result of BKSVD without SAC


% initialize dictionary with random columns of norm 1
D1 = randn(size(Y,1),K);
D1 = D1 ./  repmat(sqrt(sum(D1.^2)),size(Y,1),1);
D2 = D1;
D3 = D1;

%First we run KSVD (as initialization to our algorithm and for comparison)
X1 = []; C1 = [];
h=waitbar(0,'K-SVD');
for i = 1:max_it
    %block sparse coding
    %k*s-sparse reprentations of Y w.r.t D1 are calculated
    %(d0= [1,2,...,K] means sparse and not block sparse)
    %1 means X1 is explicitly calculated
    [X1 C1] = simult_sparse_coding(D1,Y,d0,k*s,1);
    %KSVD - updates every atom in D1 and nonzero values in X1 to minimize representation error
    [X1 D1] = KSVD_(X1, D1, Y, d0, C1);
    waitbar(i/max_it);
    % KSVD method and BKSVD method use same number of iterations:
%     if (i==max_it)
%         D2=D1; % save KSVD results after half the iterations
%     end
end
close(h);


%Run BKSVD+SAC
h=waitbar(0,'BK-SVD & SAC');
for i = 1:max_it
    %ks-sparse reprentations of Y w.r.t D2 are calculated
    %(0 means only C2s is calculated, but not X2s, this is all we need for SAC)
    %C2s of size LxK contains true at (i,j) if X0(j,i)!=0
    [X2s C2s] = simult_sparse_coding(D2,Y,d0,k*s,0);
    %SAC-finds block structure d2 given C2s and max block size s
    d2 = sparse_agg_clustering(C2s, s);
    %k-block-sparse reprentations X2 over d2 of Y w.r.t D2 are calculated
    %C2 of size LxB contains true at (i,j) if X2(:,i)  'uses' block nr j
    [X2 C2] = simult_sparse_coding(D2,Y,d2,k,1);
    %KSVD - updates every atom in D2 and nonzero values in X2 to minimize representation error
    [X2 D2] = KSVD_(X2, D2, Y, d2, C2);
    waitbar(i/max_it);
end
close(h);

%Run BKSVD without SAC
if nargout > 3
    h=waitbar(0,'BK-SVD');
    for i = 1:max_it
        %k-block-sparse reprentations X2 over d0 of Y w.r.t D3 are calculated
        [X3 C3] = simult_sparse_coding(D3,Y,d0,k,1);
        %KSVD - updates every atom in D2 and nonzero values in X2 to minimize representation error
        [X3 D3] = KSVD_(X3, D3, Y, d0, C3);
        waitbar(i/max_it);
    end
    close(h);
end


EY = norm(Y,'fro');
e1 = norm(Y-D1*X1,'fro')/EY %KSVD error
e2 = norm(Y-D2*X2,'fro')/EY %BKSVD+SAC error
e3 = norm(Y-D3*X3,'fro')/EY %BKSVD error
