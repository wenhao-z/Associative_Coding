% Training an Associative Coding Model based on natural images
% Wen-Hao Zhang,
% @Carnegie Mellon University, Sep-9, 2016

% Set the working path
setWorkPath;

AssoCode.Path_DataBase = Path_DataBase;
AssoCode.Path_RootDir = Path_RootDir;

% Load model parameters
defaultParsAssoCode; % default parameters sets
parsAssoCode_HC9_RF9; % particular parameters

%% Initialize model parameters 
AssoCode = parseMdlPars(AssoCode);
AssoCode = initAssoCode(AssoCode);

%% Preprocessing to generate the feedforward inputs to Boltzmann machine
% 1) sample image patches from images in a database
% 2) Using Gabor wavelets to filter image patches
% 3) Apply outputs of Gabor wavelets into nonlinearity to maximize entropy
[InputBM, AssoCode] = preprosAssoCode(AssoCode);

%% Train a Boltzmann machine
%fileName = fullfile(Path_RootDir, 'BatchDataSet', 'HC9_RF9_SF3', 'trainBMPars(21-Dec).mat') ;
%load(fileName);
AssoCode = mfFitBM(InputBM, AssoCode);
