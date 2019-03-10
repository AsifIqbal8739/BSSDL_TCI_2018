
function [P,PX,PY,S,SX,SY] = extract_patches_from_image(ima,nPatches,PATCH,JITTER,sDim)


P = [];   % patch repository
S = [];   % patch subspace repository
[rows,cols] = size(ima);

[JX,JY] = meshgrid([-JITTER.COLS*JITTER.NUM:JITTER.COLS:JITTER.COLS*JITTER.NUM],[-JITTER.ROWS*JITTER.NUM:JITTER.ROWS:JITTER.ROWS*JITTER.NUM]);
JX = JX(:);
JY = JY(:);
nJitter = length(JX);
halfPatchWidth = floor(PATCH.COLS/2);
halfPatchHeight = floor(PATCH.ROWS/2);

%% select random patches from ima
% the straightforward way for coding this would be to select random
% positions and then loop over them and exrtact. This could be rather slow.
% Instead we use im2col to obtain all the patches from the image, and then
% we extract the patches.

% patches must be far away enough from boundary (for jittering)
% here we select patches by their top left corner
min_row = JITTER.ROWS*JITTER.NUM+1;
max_row = rows - JITTER.ROWS*JITTER.NUM - PATCH.ROWS +1;
min_col = JITTER.COLS*JITTER.NUM+1;
max_col = cols - JITTER.COLS*JITTER.NUM - PATCH.COLS+1;
select_row = randint(1,nPatches,[min_row,max_row]); % select legal rows randomly
select_col = randint(1,nPatches,[min_col,max_col]); % select legal cols randomly

% total indices range is that used later by im2col
select_ind = sub2ind([rows-PATCH.ROWS+1,cols-PATCH.COLS+1],select_row,select_col);

% all patches 
ALL_PATCHES = im2col(ima,[PATCH.ROWS PATCH.COLS],'sliding');

% we want to record the center position of each patch
% build grids of center pixel positions corresponding to the patches in
% all_patches
[center_col,center_row] = meshgrid([halfPatchWidth+1:cols-halfPatchWidth],[halfPatchHeight+1:rows-halfPatchHeight]);
center_col = center_col(:);
center_row = center_row(:);

% extract patches and their center position
P = zeros(PATCH.ROWS*PATCH.COLS, nJitter,nPatches); 
PX = zeros(nJitter,nPatches);
PY = zeros(nJitter,nPatches);
for i=1:nJitter,
    curr_ind = select_ind + JY(i) + JX(i)*(rows-PATCH.ROWS+1);   % indices into the all_patches array
    P(:,i,:) = ALL_PATCHES(:,curr_ind);    
    PX(i,:) = center_col(curr_ind);
    PY(i,:) = center_row(curr_ind);
end
% fit subspaces
S = zeros(PATCH.ROWS*PATCH.COLS, sDim,nPatches);
SX = PX(ceil(nJitter/2),:);
SY = PY(ceil(nJitter/2),:);
for i=1:nPatches,
%     tmpPatches = double(P(:,:,i));
%     tmpPatches = tmpPatches - repmat(mean(tmpPatches,2),1,size(u,2));
%     [u,s,v] = svds(double(tmpPatches,sDim)); % obtain orthonormal basis
    
    [u,s,v] = svds(double(P(:,:,i)),sDim); % obtain orthonormal basis

    S(:,1:size(u,2),i) = u;
end

% reshape patches to a 2D array and coords into 1D array
P = reshape(P,size(P,1), nJitter*nPatches);
PX = reshape(PX,1,numel(PX));
PY = reshape(PY,1,numel(PY));
% reshape subspace to a 2D array
S = reshape(S,size(S,1), sDim*nPatches);















