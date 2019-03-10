function b = block_sparsity(X, d, th)

b = 0; L = size(X,2);
for i = 1:max(d)
    b = b+sum(sum((X(d==i,:)).^2)>th);
end
b = b/L;