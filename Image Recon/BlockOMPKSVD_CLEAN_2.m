% Block OMP + Block KSVD Complete Code
function [DBK,XBK,EE_BK] = BlockOMPKSVD_CLEAN_2(Y_,DBK,k,d_ini,B,noIt,gg,Y_Orig)
if ~exist('gg','var');    gg = 0; end
N = size(Y_,2); s = size(DBK,2)/B;
[EE_BK,avgBlockBK] = deal(zeros(1,noIt));
for i = 1:noIt
%     [XBK,CatomBK] = BlockOMP(Y_,DBK,k,d_ini,B);
    [XBK,CatomBK] = wrapp(DBK,Y_,d_ini,k,1);
    [DBK,XBK,EE_BK(i)] = BlockKSVD(Y_,DBK,XBK,CatomBK,d_ini,B,0); 
    if gg == 1        
        avgBlockBK(i) = nnz(XBK)/(s*N);        
        EE_BK(i) = DispError(Y_Orig,DBK,XBK,0);
        fprintf('Iteration: %3d, Avg Blocks: %0.2f, Error: %0.4f\n',i,avgBlockBK(i),EE_BK(i));
    end
end

end

% wrapper
function [XBK,CatomBK] = wrapp(DBK,Y_,d_ini,k,gg)
    [XBK,Catom] = simult_sparse_coding(DBK,Y_,d_ini,k,gg);
    temp = mod(find(Catom'),size(Catom,2));
    CatomBK = reshape(temp,k,size(Y_,2));
end