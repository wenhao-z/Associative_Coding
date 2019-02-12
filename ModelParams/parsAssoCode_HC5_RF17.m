% The parameters for associative coding model.

% Wen-Hao Zhang, Sep-5, 2016
% @Carnegie Mellon University


%% Parameters of raw image input from database (Layer0)

% Get the files in database
dirNow = cd(Path_DataBase);
Layer0.fileList = dir('imk*');
cd(dirNow);
clear dirNow

% Van Hateran's image database
Layer0.sizeRawImage = [1024, 1536];
Layer0.ImgPatch = [];

% Parameters of sampled image patches
Layer0.sizeImagePatch = [49, 49]; % raw image input (another option is 48X48)
%                                 This needs to in accordance with
%                                 Layer1.sizeKerl, Layer1.intNeighbKerl and Layer2.numHyperCol
Layer0.numImgPatchPerImg = 40; % number of image patches sampled from each raw image in database

Layer0.bWhiten = 1; % 1: whitening image pathces; 0: no;

% The random seed for randomly sampling image patches
Layer0.seedSampleImg = round(sum(clock)*1e3);

% The random seed for shuffling image patches
Layer0.seedShuffleImgPatch = round(sum(clock)*5e2);

Layer0.bSaveImgPatch = 1; % 1: save image patches; 0: do not save;

Layer0.saveName = 'TrainDatSet';
% May consider other image database afterwards

%% Parameters of encoding model (Layer1)

% Gabor wavelets
Layer1.sizeKerl = [17, 17]; % size of 2D Gabor wavelets. An ODD number is STRONGLY recommended to avoid numerical error.
Layer1.intNeighbKerl = (Layer1.sizeKerl+1)/2; % displacement between neighbor kernels

Layer1.numOrient = 16; % number of orientation of wavelates
Layer1.numSpatFreq = 1; % number of spatial frequency of wavelates,
% need to be optimized
Layer1.OrientArray = linspace(-pi/2, pi/2, Layer1.numOrient+1);
Layer1.OrientArray(1) = [];
Layer1.SpatFreqArray = 4;

Layer1.kOctave = 2.5; % frequency bandwidth

% Parameters of nonlinearity
Layer1.depthNonlienarity = 12; % 2^DepthNonlienarity to discretize the outputs of Gabor wavelets
Layer1.meanFiringRate = 0.5; % unit: hz/ time bin;

% May consider other encoding model afterwards, e.g., sparse coding

%% Parameters of Boltzmann machine (Layer2)
Layer2.numHyperCol = [9, 9]; % number of hypercolumns

% The number of visible and hidden neurons are EXACTLY the same.
Layer2.numVisNeuron = prod(Layer2.numHyperCol)...
    * Layer1.numOrient * Layer1.numSpatFreq;
Layer2.numHiddenNeuron = Layer2.numVisNeuron;

% Bool variables to specify which connections to be learnt in BM
Layer2.bLearnW22 = 1; % 1: learn; 0: do not learn
Layer2.bLearnW11 = 0;
Layer2.bLearnW12 = 0;

% Whether connection matrices have lateral connections (off-diagonal terms)
Layer2.bLateralW22 = 1;
Layer2.bLateralW11 = 0;
if Layer2.numHiddenNeuron == Layer2.numVisNeuron
    Layer2.bDiagW12 = 1; % one to one feedforward connections from Visible to Hidden neurons
else
    Layer2.bDiagW12 = 0;
end

% Initialize un-learned connection matrix
if ~Layer2.bLearnW11
%     Layer2.W11 = 0.5 * eye(Layer2.numVisNeuron);
    Layer2.W11 = zeros(Layer2.numVisNeuron);
end

if ~Layer2.bLearnW12
    if Layer2.bDiagW12
        Layer2.W12 = 0.02 * ones(1, Layer2.numVisNeuron);
    else
        Layer2.W12 = 0.5 * rand(Layer2.numVisNeuron, Layer2.numHiddenNeuron);
    end
end

Layer2.bNonlinearity = 1; % 1: a nonlinearity set according to the cumulative density function; 
                          % 0: doesn't use nonlinearity.

Layer2.bSpkGenerate = 0; % 1: Generate spike trains as input; 0: using mean firing rate

if Layer2.bSpkGenerate
    Layer2.seedSpkGenerator = 0; % random seed for generating spikes
    Layer2.nSpkPerImgPatch = 20; % number of spikes generated under each image patch
end

% Copy some parameters of encoding model
Layer2.numOrient = Layer1.numOrient; % number of orientation of wavelates
Layer2.numSpatFreq = Layer1.numSpatFreq; % number of spatial frequency of wavelates,
Layer2.OrientArray = Layer1.OrientArray;
Layer2.SpatFreqArray = Layer1.SpatFreqArray;

%%
% ------------------------
% Optimization parameters
% ------------------------
Layer2.maxEpoch = 1e3; % maximal number of epoches to train a BM

Layer2.szBatch = 100; % number of samples in a batch

% Number of baches of samples in an epoch
Layer2.numBatchPerEpoch = ceil(Layer0.numImgPatchPerImg * length(Layer0.fileList) ...
    / Layer2.szBatch);

Layer2.learnRate = 0.2; % Learning rate

Layer2.lambdaL2RegW22 = 0; % Weight decay (L2 regularization) 1e-2

Layer2.lambdaMomentum = 0.8; % Momentum

% Parameters for iterative dynamics of mean field Boltzmann machine
Layer2.coefIter = 0.8; % used to updating X by using Eq: X + coefIter * dX;
Layer2.maxIter = 5; % maximal number of iteraction for mean field dynamics
Layer2.tolDX = 1e-7; % stopping criteria of mean field dynamics

Layer2.maxNumCD = 1; % number of steps of Contrastive Divergence

Layer2.nEpochtoSave = 1;

Layer2.saveName = 'trainBMPars';

Layer2.bPlotinLearn = 1;

% Likelihood value

% Training set/ validating set/ testing set


%% Fold the parameters of all layers
AssoCode.Layer0 = orderfields(Layer0);
AssoCode.Layer1 = orderfields(Layer1);
AssoCode.Layer2 = orderfields(Layer2);

clear Layer0 Layer1 Layer2
