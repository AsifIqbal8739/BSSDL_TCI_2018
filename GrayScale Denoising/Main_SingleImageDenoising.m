% Code for image denoising
clc; clear all; close all;
addpath '..\Helper_Folders\Images';
addpath '..\Helper_Folders\FRIST_ivp2017- Wen2017';
addpath '..\Helper_Folders\FRIST_ivp2017- Wen2017\FRIST_tool';
addpath '..\Helper_Folders\BM3D';
addpath '..\Helper_Folders\ksvdbox13';
addpath '..\Helper_Folders\Manor-2012-Block-Sparse-Rep';
addpath '..\Helper_Folders\ompbox10';

%% Data Setup
pSize = 7;  % patch size
n = pSize^2;    K = 255;     s = 3;  k = [2,2,2,3];      % Sig_dim, nAtoms,  Atoms/block, block-sparsity
Lambdas = [0.04,0.03,0.04,0.025];   % for BSSDL
noIt = 10;         % Dict Learning Iterations  
B = K/s;          % Number of blocks in D. Should be an order of magnitude higher than the 
d_ini = repmat(1:B, s,1); d_ini = d_ini(:)';  %block structure with B blocks of size s
d0 = 1:K;
Sigma = 15;
Methods = {'FRIST','KSVD','BM3D','BKSVD','BSSDL'};
ImNames = {'lena.png','boat.png','barbara.png','baboon.bmp'};
[Denoised,TimeT] = deal(cell(4,5));
PSNR_Denoised = zeros(4,5);
Dicts = cell(3);        % KSVD, BKSVD, BSSDL
%% Lets go
for ImS = 1:length(ImNames)
    fprintf('Image: %s\n',ImNames{ImS});
    Im_O = double(imread(ImNames{ImS}));
    if length(size(Im_O)) ~= 2
        Im_O = rgb2gray(Im_O);
    end
    Noise = Sigma * randn(size(Im_O));
    % FRIST 
        Im_N = Im_O + Noise;
        tic;
        [denoised, transform, outputParam] = FRIST_imagedenoising_WRAPPER(Im_O,Im_N, ...
                        noIt, pSize^2,Sigma);
        TimeT{ImS,1} = toc;
        Denoised{ImS,1} = denoised;
        PSNR_Denoised(ImS,1) = outputParam.psnrOut;
        fprintf('%s Done \t',Methods{1});
    % KSVD
        tic;
        [imout, dict] = ksvddenoise_WRAPPER(Im_N, noIt, pSize, Sigma);
        TimeT{ImS,2} = toc;
        Denoised{ImS,2} = imout;
        PSNR_Denoised(ImS,2) = psnr(imout,Im_O,255);
        Dicts{ImS,1} = dict;
        fprintf('%s Done \t',Methods{2});
    % using 0-1 intensity
    Im_O = im2double(imread(ImNames{ImS}));
    Im_N = Im_O + Noise./255;
    % BM3D
        tic;
        [PSNR, y_est] = BM3D(Im_O, Im_N, Sigma, 'np', 0);
        TimeT{ImS,3} = toc;
        Denoised{ImS,3} = y_est;
        PSNR_Denoised(ImS,3) = psnr(y_est,Im_O,255);
        fprintf('%s Done \t',Methods{3});
        
    [Im_Data,Im_O_Loc] = pic2patches(Im_N,pSize);     % Color Patches together OVERLAPPING
    MeanVec = mean(Im_Data,1);
    Im_Data_Centered = bsxfun(@minus,Im_Data,MeanVec);    
        
    % BKSVD
        DBK = normc(randn(n,K));
        tic;
        for it = 1:noIt
            [X1s, C1s] = simult_sparse_coding(DBK,Im_Data_Centered,d0,k(ImS)*s,0);
            d1 = sparse_agg_clustering(C1s, s);
        %     d1 = d_ini;
            [XBK, C1] = simult_sparse_coding(DBK,Im_Data_Centered,d1,k(ImS),1);
            [XBK, DBK] = KSVD_(XBK, DBK, Im_Data_Centered, d1, C1);
        end
        TimeT{ImS,4} = toc;
        Im_P_Own = repmat(MeanVec,n,1) + DBK*XBK;
        [Im_Rec_Own] = patches2pic(Im_P_Own,Im_O_Loc,pSize);
        Denoised{ImS,4} = Im_Rec_Own;
        PSNR_Denoised(ImS,4) = psnr(Im_Rec_Own,Im_O);
        Dicts{ImS,2} = DBK;
        fprintf('%s Done \t',Methods{4});
    % BSSDL
        % K-Means Initial Clustering
        tic;
		nClust = B;
		opts = statset('Display','off','UseParallel',0);
		[IndY,CentY] = kmeans(Im_Data_Centered',nClust,'Distance','cityblock','Replicates',2,'Options',opts);

		% Dictionary Initialization
		Dini = zeros(n,K);     
		% Generating s atoms for each group
		for i = 1:B
			temp = Im_Data_Centered(:,IndY == i);
			[dd,~] = eig(temp*temp');
			tt = (i-1)*s+1:i*s;
			Dini(:,tt) = dd(:,end:-1:end-s+1);
        end
        [DKM,XKM,EKM] = BSSDL_Multi_CLEAN(Im_Data_Centered,Dini,d_ini,Lambdas(ImS),noIt,10,0);
%         [DKM,XKM,EKM] = BSSDL_Multi_CLEAN(Im_Data_Centered,Dini,d_ini,0.05,noIt,10,0);
        TimeT{ImS,5} = toc;
        Im_P_Own = repmat(MeanVec,n,1) + DKM*XKM;
        [Im_Rec_Own] = patches2pic(Im_P_Own,Im_O_Loc,pSize);
        Denoised{ImS,5} = Im_Rec_Own;
        PSNR_Denoised(ImS,5) = psnr(Im_Rec_Own,Im_O);
        Dicts{ImS,3} = DKM;
        fprintf('%s Done \n',Methods{5});
end
% save('DenoisedImages2.mat', 'Denoised', 'PSNR_Denoised', 'Dicts', 'TimeT');

