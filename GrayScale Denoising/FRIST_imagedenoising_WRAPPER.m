% wrapper for FRIST
function [denoised, transform, outputParam] = FRIST_imagedenoising_WRAPPER(Im_O,Im_N, ...
                    noIt, n,Sig)
               %% info transfer
    param.n = n;
    param.sig = Sig;
    param.stride = 1;           % fully overlapped
    param.iter = noIt;
    data.noisy = Im_N;
    data.oracle = Im_O;
    param.isKmeansInitialization = false;
    param.numBlock = 4;         % 4 rotation directions
    [denoised, transform, outputParam] = FRIST_imagedenoising(data, param);    
                
end