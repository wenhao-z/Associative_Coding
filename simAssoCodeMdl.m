% Simulate whole associative model
% Wen-Hao Zhang, Nov-4, 2016
% wenhaoz1@andrew.cmu.edu
% @Carnegie Mellon University

% Set the working path
setWorkPath;
addpath(fullfile(Path_RootDir, 'simExp'));
addpath(fullfile(Path_RootDir, 'simExp', 'taskScript'));

%% Load trained models and set parameters
folderName = 'HC9_RF9_SF3';
% mdlFileName = 'trainBMPars(21-Dec)L2Reg1e-5.mat';
mdlFileName = 'trainBMPars_170609_1742.mat';

% AssoCode.Layer2.flagNonLincFunc = 2;
% folderName = 'HC9_RF9_SF4';
% mdlFileName = 'trainBMPars(28-Sep).mat';

% Load trained model
fileName = fullfile(Path_RootDir, 'BatchDataSet', folderName, ...
    mdlFileName);
load(fileName);

% Some particular parameters
AssoCode.Layer2.maxIter = 1e2;
AssoCode.Layer2.tolDX = 1e-15;
AssoCode.Layer2.coefIter = 0.05;

%% Generate nonlinearity 
if ~isfield(AssoCode.Layer1, 'nlFunc_Edge') && (AssoCode.Layer1.flagNonLincFunc ~=0)
    % Load the nonlinearity from filterImgPatch 
    % Nonlinearity is set as the cdf. of filter filterImgPatch 
    
    AssoCode = parseMdlPars(AssoCode);    
    load(fullfile(Path_RootDir, 'BatchDataSet', folderName, ...
        [AssoCode.Layer0.fileName, '.mat']), 'filterImgPatch');

    
    [~, nlFunc_Edge] = nonlinearFunc(filterImgPatch, AssoCode.Layer1);
    AssoCode.Layer1.nlFunc_Edge = nlFunc_Edge;
end
clear IntBins Edge
    
% AssoCode.Layer0.bWhiten = 0;
%% Perform virtual experiments
% The parameters of each task is defined separately in runTask script.

flagTask = 3;
% 1. A single bar with different length
% 2. Response wrt. the size and contrast of grating with optimal orientation
% 3. Orientation tuning wrt. a single grating in cRF under different contrast
% 4. Two concentric gratings with varying the orientation of outter grating
% 5. Response under different bar background
% 6. Two surrounding bars with the same orientation but varying horizontal and vertical distance

switch flagTask
    case 1
        % Bar with different length (end-stopping)
        runTask1;
    case 2
        % Neural response wrt. the size and contrast of grating with optimal orientation
        runTask2;
    case 3
        % Tuning curve wrt. a single grating in cRF under different contrast
        runTask3;
    case 4
        % Two gratings
        runTask4;
    case 5 
        % Response under different bar background
        runTask5;
    case 6
        % Two surrounding bars with the same orientation but varying
        % horizontal and vertical distance
        runTask6;
    case 7
        % A temporal evolution of network activities with time
        runTask7;
end
