% Synthetic Data Simulation for block sparse signals
clc; clear all; close all;
addpath '..\Helper_Folders\Manor-2012-Block-Sparse-Rep';

%% Data Stuff
n = 50;  K = 120;  N = 2000; s = 3;        % Sig_dim, nAtoms, nSig, Atoms/block, 
noIt = 30;              % Dict Learning Iterations  
B = K/s;                % Number of blocks in D. Should be an order of magnitude higher than the 
                        % block sparsity k, otherwise the representations wont be sparse
% d0 = 1:K;             % block structure with K blocks of size 1 (i.e. block structure ignored)
d_ini = repmat(1:B, s,1); d_ini = d_ini(:)';  %block structure with B blocks of size s

SnRdB_ = [-10, -5, 0, 5];
BlockSpar = [2, 3, 4, 5];
SS = 4; SnRdB = SnRdB_(SS);            % Select Noise level here
BS = 3; k = BlockSpar(BS);             % Select Block Sparsity here

D_Orig = normc(randn(n,K));
X_Orig = zeros(K,N);    % Coefficients for Representation
d_Used = zeros(k,N);    % Blocks used for each signal

%% Block Orthonormalize the Original Dictionary
% Can be skipped if need be
for b = 1:B
    gg = (b-1)*s+1:b*s;
    [tt,hh] = qr(D_Orig(:,gg));
    D_Orig(:,gg) = tt(:,1:s);
end

%% Block-Sparse Signal Generation
for nSig = 1:N
    pp = randperm(B,k); % Which blocks to use for signal generation
    nInd = [];
    for ll = 1:k
        nInd = [nInd;(d_ini == pp(ll)')];
    end
    nInd = any(nInd);    
    X_Orig(nInd,nSig) = rand(sum(nInd),1) - 0.5;   
    d_Used(:,nSig) = pp;
end
Y = (D_Orig*X_Orig);  % Block-Sparse Signals

Yn = awgn(Y,SnRdB,'measured');      % Noisy Signal Matrix
Dini = normc(randn(n,K));

%% Using Block OMP and Block KSVD
DBK = Dini;                 
tic; [DBK,XBK,EE_BK] = BlockOMPKSVD_CLEAN_2(Yn,DBK,k,d_ini,B,noIt,1,Y);   secs2hms(toc)
% figure(1); plot(EE_BK); title(sprintf('BKSVD Error')); hold on;
EE_BK_Final = DispError(Y,DBK,XBK,0);

%% Call your algorithm here

Lambdas = [ 0.7200    0.8200    0.8800    0.9000;
            0.2600    0.3100    0.3300    0.3300;
            0.1300    0.1400    0.1400    0.1500;
            0.0800    0.0900    0.0900    0.0860];

lambda = Lambdas(SS,BS);
% Dk = normc(randn(n,K));
Dk = Dini;
tic; [Dk,Xk,Ek,AvgB] = BSSDL(Yn,Dk,d_ini,lambda,noIt,10,Y); secs2hms(toc)
Error_K = DispError(Y,Dk,Xk,1);   % Final Representation Error Normalized
% figure(2); plot(Ek); title(sprintf('BSSDL Error'));
