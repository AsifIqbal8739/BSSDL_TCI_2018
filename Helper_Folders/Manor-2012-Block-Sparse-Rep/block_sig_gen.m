function [X, nzer_b] = block_sig_gen(K, L, d, k)

%written by Kevin Rosenblum, 2007-2008
%kevin@technion.ac.il

X = zeros(K, L);
p = randperm(max(d));
nzer_b = p(1:k);

dk = zeros(1,K);
for  n = 1:k
    dk = dk | (d==nzer_b(n));
end    

X(dk,:) = rand(sum(dk),L)-0.5;
