% Wrapper for KSVD Denoise function
function [imout, dict] = ksvddenoise_WRAPPER(Im_N,noIt,pSize,Sigma)
    params.x = Im_N;
    params.blocksize = pSize;
    params.dictsize = 256;
    params.sigma = Sigma;
    params.maxval = 255;
    params.trainnum = 40000;
    params.iternum = noIt;
    params.memusage = 'high';
    params.exact = 1;
    [imout, dict] = ksvddenoise(params,0);
end