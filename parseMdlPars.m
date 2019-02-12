function AssoCode = parseMdlPars(AssoCode)
% Recalculate some model parameters if the parameters are updated

% Wen-Hao Zhang, Oct-4, 2016
% @Carnegie Mellon University

%% Update the path
setWorkPath;
AssoCode.Path_DataBase = Path_DataBase;
AssoCode.Path_RootDir = Path_RootDir;

%% Parameters of encoding model (Layer1)
AssoCode.Layer1.intNeighbKerl   = (AssoCode.Layer1.sizeKerl+1)/2; % displacement between neighbor kernels

%% Parameters of Boltzmann machine (Layer2)
AssoCode.Layer2.numVisNeuron    = prod(AssoCode.Layer2.numHyperCol)* AssoCode.Layer1.numOrient * AssoCode.Layer1.numSpatFreq; % number of  visible neuron
AssoCode.Layer2.numHiddenNeuron = AssoCode.Layer2.numVisNeuron; % The number of visible and hidden neurons are EXACTLY the same.

%% Parameters of the input
AssoCode.Layer0.sizeImagePatch = AssoCode.Layer1.sizeKerl ... 
    + (AssoCode.Layer2.numHyperCol-1).*AssoCode.Layer1.intNeighbKerl;

%% Optimization parameters
AssoCode.Layer2.numBatchPerEpoch = ceil(AssoCode.Layer0.numImgPatchPerImg * length(AssoCode.Layer0.fileList)/ AssoCode.Layer2.szBatch); % Number of baches of samples in an epoch

%% Parameters of save
AssoCode.parFoldName    = sprintf('HC%d_RF%d_SF%d', AssoCode.Layer2.numHyperCol(1), AssoCode.Layer1.sizeKerl(1), AssoCode.Layer1.SpatFreqArray);
AssoCode.savePath       = fullfile(AssoCode.Path_RootDir, 'BatchDataSet', AssoCode.parFoldName);

if isempty(strfind(AssoCode.Layer0.saveName, AssoCode.parFoldName))
    AssoCode.Layer0.fileName = [AssoCode.Layer0.saveName, '_', AssoCode.parFoldName];
end