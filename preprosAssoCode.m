function [InputBM, AssoCode] = preprosAssoCode(AssoCode)
% Preprocess the model to to get the inputs fed to a Boltzmann machine
% This preprocessing has two sub-process
% 1) filter image patches by using Gabor wavelets
% 2) apply the outputs of Gabor wavelets into a nonlinearity
%    (cumulative density function of Gabor's output to achieve MAX ENTROPY)
% The output of the nonlinearity is exactly the inputs of a Boltzmann machine

% Wen-Hao Zhang, Sep-12, 2016
% @Carnegie Mellon University

Path_DataBase = AssoCode.Path_DataBase;

Layer0 = AssoCode.Layer0;
Layer1 = AssoCode.Layer1;
Layer2 = AssoCode.Layer2;


%% Initialize sampled image patches (Layer 0)
% if ~(exist('BatchDataSet', 'dir') == 7)
%     mkdir(fullfile(AssoCode.Path_RootDir, 'BatchDataSet'));
% end
% dirNow = cd(fullfile(AssoCode.Path_RootDir, 'BatchDataSet'));
if ~(exist(AssoCode.savePath, 'dir') == 7)
    mkdir(AssoCode.savePath);
end
dirNow = cd(AssoCode.savePath);
fileList = dir([Layer0.fileName, '.mat']);

if isempty(fileList)
    ImgPatch = sampleImgPatch(Layer0, Path_DataBase); % [Height, Width, numImgPatchPerImg, numImg]
    
    %     % Load images into GPU
    %     if Layer2.bGPU
    %         ImgPatch = single(ImgPatch);
    %         ImgPatch = gpuArray(ImgPatch);
    %     end
    
    szImgPatch = size(ImgPatch);
    ImgPatch = reshape(ImgPatch, szImgPatch(1), szImgPatch(2), []); % Reshape into a 3D array
    
    % Rescale data range
    ImgPatch = ImgPatch/max(ImgPatch(:));    
        
    % --------------------
    % Whitening images
    % --------------------
    if Layer0.bWhiten
        lenSet = 2e4;
        nSet = ceil(size(ImgPatch, 3)/ lenSet);
        for iterSet = 1: nSet
            Idx = [(iterSet-1) *lenSet + 1, min(iterSet * lenSet, size(ImgPatch, 3))];
            ImgPatch(:,:,Idx(1):Idx(2)) = whitenImage(ImgPatch(:,:,Idx(1):Idx(2)));
        end
    end
    
    % ----------------------
    % Contrast normalization
    % ----------------------
    ImgPatch = reshape(ImgPatch, prod(szImgPatch(1:2)), []);
    ImgPatch = bsxfun(@minus, ImgPatch, mean(ImgPatch, 1)); % Per-example mean subtraction
    ImgPatch = bsxfun(@rdivide, ImgPatch, std(ImgPatch, [], 1)); % Make each image patch to be unit variance
    ImgPatch = bsxfun(@minus, ImgPatch, mean(ImgPatch, 1)); % Redo mean subtraction again to again numerical error
    ImgPatch = reshape(ImgPatch, szImgPatch(1), szImgPatch(2), []);
    
    % Shuffle the order of image patches;
    s = RandStream('mt19937ar','Seed', Layer0.seedShuffleImgPatch);
    RandStream.setGlobalStream(s);
    IdxRand = randperm(size(ImgPatch, 3));
    ImgPatch = ImgPatch(:, :, IdxRand);
    
    if Layer0.bSaveImgPatch
        % Save model parameters and image patches
        fprintf('Saving image patches...')
        if isa(ImgPatch, 'gpuArray')
            ImgPatch = double(gather(ImgPatch));
        end
        save(Layer0.fileName, 'ImgPatch', 'AssoCode','-v7.3');
        
        fprintf('Done.\n');
    end
    
    %% Filter image patches by using Gabor wavelets
    fprintf('Filter image pathces by using Gabor wavelets......');
    tStart = clock;
    % filterImgPatch = simGaborWavelets(ImgPatch, AssoCode.Layer1);
    
    % A sloppy method to overcome possible limited Memory for convolution
    lenSet = 2e4;
    nSet = ceil(size(ImgPatch, 3)/ lenSet);
    
    filterImgPatch = zeros(prod(Layer2.numHyperCol) * numel(Layer1.gaborArray), ...
        size(ImgPatch,3) );
    for iterSet = 1: nSet
        Idx = [(iterSet-1) *lenSet + 1, min(iterSet * lenSet, size(ImgPatch, 3))];
        filterImgPatch(:, Idx(1): Idx(2)) = simGaborWavelets(ImgPatch(:,:, Idx(1): Idx(2)), Layer1);
    end
    
    tEnd = clock;
    fprintf('Done!\n');
    fprintf('It takes %d seconds\n', etime(tEnd, tStart));
    
    save(Layer0.fileName, 'filterImgPatch', '-append');
    
    clear lenSet nSet iterSet
else
    % Check whether it already has Gabor's output
    varName = whos('-file', Layer0.fileName);
    varName = {varName.name};
    varName = cellfun(@(x) strfind(x, 'filterImgPatch'), varName, 'uniformout', 0);
    varName = cellfun(@(x) x==1, varName, 'uniformout', 0);
    varName = sum(cell2mat(varName));
    
    if varName
        fprintf('Loading filtered image pathces...')
        load(Layer0.fileName, 'filterImgPatch');
        fprintf('Done!\n')
    end
end
fprintf('=================================================================\n')
clear fileList loadStruct

%% Set the nonlinearity after Gabor wavelets
% The nonlinearity is the cumulative density function of each channel (row);
fprintf('Set a nonlinearity after Gabor filters...\n')

[InputBM, nlFunc_Edge] = nonlinearFunc(filterImgPatch, Layer1);
AssoCode.Layer1.nlFunc_Edge = nlFunc_Edge;

if Layer2.bSpkGenerate
    fprintf('Generating spike trains ...')
    s = RandStream('mt19937ar','Seed', Layer2.seedSpkGenerator);
    RandStream.setGlobalStream(s);
    
    InputBMSpk = zeros([size(InputBM), Layer2.nSpkPerImgPatch]);
    for iter = 1: Layer2.nSpkPerImgPatch
        InputBMSpk(:,:, iter) = (InputBM > rand(size(InputBM)));
    end
    InputBMSpk = reshape(InputBMSpk, size(InputBMSpk, 1), []);
    
    InputBM = InputBMSpk;
end

fprintf('Done!\n')
fprintf('=================================================================\n')

cd(dirNow)

clear szInputBM dirNow Idx iter iterFile szInputBM

end