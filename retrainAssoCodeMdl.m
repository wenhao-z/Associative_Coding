% Training an Associative Coding Model based on natural images
% Wen-Hao Zhang,
% @Carnegie Mellon University, Sep-9, 2016

% Set the working path
setWorkPath;

AssoCode.Path_DataBase = Path_DataBase;
AssoCode.Path_RootDir = Path_RootDir;

% Load model parameters
% defaultParsAssoCode; % default parameters sets
% parsAssoCode_HC9_RF9; % particular parameters

% Load previous models
fileName = fullfile(Path_RootDir, 'BatchDataSet', 'HC9_RF9_SF3', ...
    'trainBMPars(21-Dec)L2Reg1e-5.mat') ;
load(fileName);
AssoCode.Layer2.lambdaL2RegW22 = 1e-5;
AssoCode.Layer2.maxEpoch = 3000;
AssoCode.Layer2.ReConstError = [AssoCode.Layer2.ReConstError, ...
    zeros(1, AssoCode.Layer2.maxEpoch - length(AssoCode.Layer2.ReConstError))];
% AssoCode.Layer2.saveName = [AssoCode.Layer2.saveName, 'L2Reg1e-5'];

%% Preprocessing to generate the feedforward inputs to Boltzmann machine
% 1) sample image patches from images in a database
% 2) Using Gabor wavelets to filter image patches
% 3) Apply outputs of Gabor wavelets into nonlinearity to maximize entropy
[InputBM, AssoCode] = preprosAssoCode(AssoCode);

%% Train a Boltzmann machine
AssoCode = mfFitBM(InputBM, AssoCode);
