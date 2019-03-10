% Script to generate the final multi trial NRE results
clc; clear all; close all;
addpath '..\Helper_Folders\Images';
%% Datastuff
n = 64;  K = 96;  s = 3;  k = [2,3];      % Sig_dim, nAtoms,  Atoms/block, block-sparsity
noIt = 50;         % Dict Learning Iterations  
B = K/s;          % Number of blocks in D. Should be an order of magnitude higher than the 
                  % block sparsity k, otherwise the representations wont be sparse
d0 = 1:K;         % block structure with K blocks of size 1 (i.e. block structure ignored)
d_ini = repmat(1:B, s,1); d_ini = d_ini(:)';  %block structure with B blocks of size s
nTrials = 10;    M = 16;
pSize = sqrt(n);
ImNames = {'baboon.bmp','barbara.png','boat.png','flinstones.png','house.png','lena.png','Cameraman512.png'};

Lambdas = [0.02, 0.035, 0.02, 0.04, 0.03, 0.04, 0.025];

%% Starting here
for Im = 1:length(ImNames)
    Im_O = im2double(imread(ImNames{Im}));
    if length(size(Im_O)) ~= 2
        Im_O = rgb2gray(Im_O);
    end
    Y = im2col(Im_O, [pSize,pSize], 'distinct'); 
    fprintf('Image: %s, Lambda: %0.3f\n',ImNames{Im},Lambdas(Im));    
    for tr = 1:nTrials
        Dini = normc(randn(n,K));
        % learn A using Sapiro's algorithm
        A = Optisens(randn(n,K),M);
        % BSSDL
        gg = 0; % display on (1) or off (0)
        tic; [DKM,XKM,EKM] = BSSDL(Y,Dini,d_ini,Lambdas(Im),noIt,10,gg,Y); 
        AvgBS(Im,tr) = nnz(XKM)/(s*size(Y,2));
        
        XKM = simult_sparse_coding(A*DKM,A*Y,d_ini,k(1),1);    % for k = 2
        NRE_A{1}(Im,tr) = DispError(Y,DKM,XKM);
        XKM = simult_sparse_coding(A*DKM,A*Y,d_ini,k(2),1);    % for k = 3
        NRE_A{2}(Im,tr) = DispError(Y,DKM,XKM);
        fprintf('Trial: %d, \t AvgBS: %0.2f, \t After_BS_2: %0.4f, \t After_BS_3: %0.4f\n',...
            tr, AvgBS(Im,tr),NRE_A{1}(Im,tr),NRE_A{2}(Im,tr));
    end
end

%% Extraction
A1 = sort(NRE_A{1},2);      A2 = sort(NRE_A{2},2);
AA1 = mean(A1(:,1:6),2);    AA2 = mean(A2(:,1:6),2);