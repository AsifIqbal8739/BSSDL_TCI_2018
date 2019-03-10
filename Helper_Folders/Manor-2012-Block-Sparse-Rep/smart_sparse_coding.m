
function [X C] = smart_sparse_coding(X0, C0, d0, p, T, t, D, Y, d, k, find_X)

if ~isempty(C0)&&(nnz(d==d0)==length(d0));
    [X(:,p(1:T)) C(p(1:T),:)] = simult_sparse_coding(D,Y(:,p(1:T)), d, k, find_X);
    if nnz(C(p(1:T),:).*C0(p(1:T),:))/nnz(C0(p(1:T),:))>=t, C = C0; X = X0;
    else [X(:,p(T+1:end)) C(p(T+1:end),:)] = simult_sparse_coding(D,Y(:,p(T+1:end)), d, k, find_X);
    end
else [X C] = simult_sparse_coding(D,Y, d, k, find_X);
end