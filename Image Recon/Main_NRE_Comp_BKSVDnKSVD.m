% Script to generate results for KSVD and BKSVD without SAC with results on
% compressed signals
clc; clear all; close all;
addpath '..\Helper_Folders\Images';
addpath '..\Helper_Folders\ompbox10';
% Folder containing the code written by the authors Manor et. al. 
addpath '..\Helper_Folders\Manor-2012-Block-Sparse-Rep';

%% Datastuff
n = 64;  K = 96;  s = 3;  k = [2,3];      % Sig_dim, nAtoms,  Atoms/block, block-sparsity
noIt = 50;         % Dict Learning Iterations  
B = K/s;          % Number of blocks in D. Should be an order of magnitude higher than the 
                  % block sparsity k, otherwise the representations wont be sparse
d0 = 1:K;         % block structure with K blocks of size 1 (i.e. block structure ignored)
d_ini = repmat(1:B, s,1); d_ini = d_ini(:)';  %block structure with B blocks of size s
nTrials = 1;    M = 16;
pSize = sqrt(n);
ImNames = {'baboon.bmp','barbara.png','boat.png','flinstones.png','house.png','lena.png','Cameraman512.png'};
% [NRE_BK,NRE_K,TimeT] = deal(zeros(length(ImNames),2,nTrials));
% NRE_A - After Compression
% NRE_B - Before Compression
%% Begin
for Im = 1:length(ImNames)
    Im_O = im2double(imread(ImNames{Im}));
    if length(size(Im_O)) ~= 2
        Im_O = rgb2gray(Im_O);
    end
    Y = im2col(Im_O, [pSize,pSize], 'distinct');
    for BS = 1:length(k)
        for tr = 1:nTrials
            fprintf('Image: %s, BS: %d, Trial: %d\n',ImNames{Im},k(BS),tr);
            Dini = normc(randn(n,K));
            % BKSVD
            gg = 0; % display on (1) or off (0)
            tic; [DBK,XBK,EE_BK] = BlockOMPKSVD_CLEAN_2(Y,Dini,k(BS),d_ini,B,noIt,gg,Y);
            TimeT{1} = toc;
            NRE_B{1}(Im,BS,tr) = DispError(Y,DBK,XBK);
            % KSVD
            tic; [D_KSVD,X_KSVD] = K_SVD(Y,Dini,noIt,s*k(BS));
            TimeT{2} = toc;
            EK = DispError(Y,D_KSVD,X_KSVD);
            NRE_B{2}(Im,BS,tr) = DispError(Y,D_KSVD,X_KSVD);
            % learn A using Sapiro's algorithm
            A = Optisens(randn(n,K),M); 
            X_KSVD = simult_sparse_coding(A*D_KSVD,A*Y,d0,k(BS)*s,1);
            XBK = simult_sparse_coding(A*DBK,A*Y,d_ini,k(BS),1);
            NRE_A{1}(Im,BS,tr) = DispError(Y,DBK,XBK);
            NRE_A{2}(Im,BS,tr) = DispError(Y,D_KSVD,X_KSVD);
            fprintf('BKSVD: %0.4f, \t KSVD: %0.4f\n',NRE_A{1}(Im,BS,tr),NRE_A{2}(Im,BS,tr));
        end
    end
end

%% Data Extraction
% AA1 = median(NRE_A{1},3);   %BKSVD
% AA2 = median(NRE_A{2},3);

AA1 = mean(NRE_A{1},3);   %BKSVD
AA2 = mean(NRE_A{2},3);