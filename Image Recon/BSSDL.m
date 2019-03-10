%% Code for Block Dictionary Learning
function [DD,XX,EE,Avg_Blocks] = BSSDL(Y,Dict,d_ini,lambda,noIt1,noIt2,gg,Y_Orig)

% d_ini = block indices
% Variable Setup
N = size(Y,2);  K = size(Dict,2);
[EE,Avg_Blocks] = deal(zeros(1,noIt1));
nB = max(d_ini);        % number of blocks in the matrix
tol = 0.01;

X = zeros(K,N);
lambda_M = zeros(nB,N);

% Ortho Normalize the Blocks of Dictionary
Dict = Block_Ortho(Dict,nB,d_ini);

%% Algo goes here   
DD = Dict;  XX = X;
[D_old,D_old2] = deal(DD);
nBSize = zeros(1,nB);   % Block Sizes
for it1 = 1:noIt1            
    for b = 1:nB                
        Indd = d_ini == b;  BSize = nnz(Indd);  
        if it1 == 1;    nBSize(b) = BSize;  end
        E_l = Y - DD*XX + DD(:,Indd)*XX(Indd,:);                

        F = DD(:,Indd)'*E_l;
        Fnorm = sum(F.*F,1).^0.5;
        lambda_M(b,:) = lambda./Fnorm;

        for it2 = 1:noIt2
            S = DD(:,Indd)' * E_l;
            Snorm = sum(S.*S,1).^0.5;
            Snorm(Snorm == 0) = 0.0001;
            XX(Indd,:) = S.*max(0,(1-((lambda_M(b,:).*sqrt(BSize))./(2*Snorm))));
            % Full sparse check
%             if nnz(XX(Indd,:)) < 5
%                 [U,S,V] = svds(E_l,BSize);
%                 DD(:,Indd) = U;
%                 XX(Indd,:) = S*V';
%                 fprintf('Empty X Row Found...\n'); break;
%             end                    
            % BSize check here
            if BSize == 1
                DD(:,Indd) = normc(E_l*XX(Indd,:)');
            else
                [U,~,V] = svds(E_l*XX(Indd,:)',BSize);  
                DD(:,Indd) = U*V'; 
            end
            delta = norm(DD(:,Indd) - D_old(:,Indd),'fro');
            if delta < tol
%                 fprintf('block %d difference < tolerence\n', b);
                break;
            end
            D_old = DD; % use this in time calucation
        end
    end
    Dconv(it1) = norm(DD - D_old2,'fro')/norm(D_old2,'fro');
    D_old2 = DD;
    EE(it1) = DispError(Y_Orig,DD,XX,0);
%     Avg_Blocks(it1) = nnz(XX)/(mean(nBSize)*N);
%     if mod(it1,5) == 0
        avgBlockBK = nnz(XX)/(mean(nBSize)*N);
        if gg == 1
            fprintf('Iteration: %3d, Avg Blocks: %0.2f, Error: %0.4f, DConv: %0.4f\n',it1,avgBlockBK,EE(it1),Dconv(it1));
        end
%     end
end
end


%% Block Orthonormalize the dictionary
function D = Block_Ortho(D,nB,d_ini)
for b = 1:nB
    s = nnz(d_ini == b);
    gg = (b-1)*s+1:b*s;
    [tt,~] = qr(D(:,gg));
    D(:,gg) = tt(:,1:s);
end

end