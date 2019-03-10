function D = block_orthonormalize(D, d)

for k =1:max(d)
  col = d==k;
  [D(:,col), R] = qr(D(:,col),0);
end