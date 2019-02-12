% Tune the parameters of Gabor wavelets to maximize their outputs

% Wen-Hao Zhang, Sep-7, 2016
% @Carnegie Mellon University

setWorkPath;
% Load model parameters
parsAssoCode;

AssoCode.Path_DataBase = Path_DataBase;
AssoCode.Path_RootDir = Path_RootDir;

Layer0 = AssoCode.Layer0;
Layer1 = AssoCode.Layer1;

%% Read the sampled image patches
if ~(exist('BatchDataSet', 'dir') == 7)
    mkdir(fullfile(AssoCode.Path_RootDir, 'BatchDataSet'));
end
dirNow = cd(fullfile(AssoCode.Path_RootDir, 'BatchDataSet'));
fileList = dir([Layer0.saveName, '*']);

if isempty(fileList)
    ImgPatch = sampleImgPatch(Layer0, Path_DataBase); % [Height, Width, numImgPatchPerImg, numImg]
    
    % reshape ImgPatch into a 3D array
    szImgPatch = size(ImgPatch);
    ImgPatch = reshape(ImgPatch, szImgPatch(1), szImgPatch(2), []);
    
    % Shuffle the order of image patches;
    s = RandStream('mt19937ar','Seed', Layer0.seedShuffleImgPatch);
    RandStream.setGlobalStream(s);
    IdxRand = randperm(size(ImgPatch, 3));
    ImgPatch = ImgPatch(:, :, IdxRand);
    
    % Save model parameters and image patches
    % Split whole ImgPatch into small BATCHES
    fprintf('Saving image patches...')
    save(Layer0.saveName, 'ImgPatch', 'AssoCode','-v7.3');
   
    fprintf('Done.\n');
else
    fprintf('Load previously processed image pathces......')
    load(Layer0.saveName, 'ImgPatch', 'AssoCode');
    fprintf('Done!\n')
end
%%
ImgPatch = ImgPatch(:,:, randperm(length(ImgPatch), 1e4));
ImgPatch = reshape(ImgPatch, Layer0.sizeImagePatch(1), Layer0.sizeImagePatch(2), []);
% ImgPatch = bsxfun(@minus, ImgPatch, mean(mean(ImgPatch, 1), 2) );

% Generate parameter grid
OmegaArray = 3:1:15;
kOctaveArray = 2.5;

[OmegaArray, kOctaveArray] = ndgrid(OmegaArray, kOctaveArray);

sumFilterImgAbs = zeros(size(OmegaArray));
%%
for iterPar = 1: numel(OmegaArray)
    fprintf('iterPar: %d/%d\n', iterPar, numel(OmegaArray));
    % Generate a family of Gabor wavelets under given parameter set
    gaborArray = gaborWavelets2D(Layer1.sizeKerl, OmegaArray(iterPar), ...
        Layer1.OrientArray, kOctaveArray(iterPar));
    
    % Flip gaborArray before using convn code in MATLAB
    gaborArray = cellfun(@(x) flip(flip(x,1), 2), gaborArray, 'uniformout', 0);
    filterImgPatch = cellfun(@(x) convn(ImgPatch, x, 'same'), gaborArray, ...
        'uniformout', 0);
    filterImgPatch = cellfun(@(x) mean(x(:).^2), filterImgPatch);
    
    sumFilterImgAbs(iterPar) = mean(filterImgPatch);
end

%% Plot the results

surf(OmegaArray, kOctaveArray/pi, sumFilterImgAbs);

xlabel('Spatial frequency')
ylabel('Bandwidth')

