function AssoCode = initAssoCode(AssoCode)
% Initialize the connections in each layer in the model and input-output
% dataset

% Wen-Hao Zhang
% @Carnegie Mellon University, Sep-9, 2016

fprintf('Initialize model...\n')
%% Extract variables from AssoCode struct
Layer1 = AssoCode.Layer1;
Layer2 = AssoCode.Layer2;

%% Initialize Gabor functions (Layer 1)
gaborArray = gaborWavelets2D(Layer1.sizeKerl, Layer1.SpatFreqArray, ...
    Layer1.OrientArray, Layer1.kOctave);

% Patch code (Jan-6, 2017)
% Normalize all Gabor filters to have the same mean and var with reference
% of 0 deg Gabor filter.
% This is to overcome numerical error of limited spatial resolution
avgKerl = mean(gaborArray{8}(:));
stdKerl = std(gaborArray{8}(:));
gaborArray = cellfun(@(x) x./std(x(:))*stdKerl, gaborArray, 'uniformout', 0);
gaborArray = cellfun(@(x) x-mean(x(:))+avgKerl, gaborArray, 'uniformout', 0);

Layer1.gaborArray = gaborArray;
AssoCode.Layer1 = Layer1;

%% Initialize the connection matrix in Boltzmann machine (Layer 2)
% For a learnt connection matrix, its initialization is rand matrix;
% otherwise, it is specified by hand.

% W22: connections between hidden units
% A symmetric matrix with all diagonal terms are zero.
if Layer2.bLateralW22
    if Layer2.bLearnW22
        Layer2.W22 = Layer2.J22*rand(Layer2.numVisNeuron); % An extra term is a bias term
    else
        Layer2.W22 = Layer2.J22*ones(Layer2.numVisNeuron);
    end
else
    Layer2.W22 = zeros(Layer2.numVisNeuron);
end
Layer2.W22 = (Layer2.W22 + Layer2.W22')/2;
Layer2.W22 = Layer2.W22 - diag(diag(Layer2.W22));

% W11: connections between visibleunits
% A symmetric matrix with all diagonal terms are zero.
if Layer2.bLateralW11
    if Layer2.bLearnW11
        Layer2.W11 = Layer2.J11*rand(Layer2.numVisNeuron); % An extra term is a bias term
    else
        Layer2.W11 = Layer2.J11*ones(Layer2.numVisNeuron);
    end
else
   Layer2.W11 = zeros(Layer2.numVisNeuron);
end
Layer2.W11 = (Layer2.W11 + Layer2.W11')/2;
Layer2.W11 = Layer2.W11 - diag(diag(Layer2.W11));

if Layer2.bLearnW12
    if Layer2.bDiagW12
        Layer2.W12 = Layer2.J12 * randn(1, Layer2.numVisNeuron);
    else
        Layer2.W12 = Layer2.J12 * randn(Layer2.numVisNeuron, Layer2.numHiddenNeuron);
    end
else
    if Layer2.bDiagW12
        Layer2.W12 = Layer2.J12 * ones(1, Layer2.numVisNeuron);
    else
        Layer2.W12 = Layer2.J12 * ones(Layer2.numVisNeuron, Layer2.numHiddenNeuron);
    end
end

Layer2.Bias1 = Layer2.Bias1_Init*ones(Layer2.numVisNeuron, 1); % Suggested by Hinton's practical guide
Layer2.Bias2 = Layer2.Bias2_Init*ones(Layer2.numHiddenNeuron, 1);

if ~gpuDeviceCount
    Layer2.bGPU = 0;
end

AssoCode.Layer2 = Layer2;

%% Save model struct
if ~(exist(AssoCode.savePath, 'dir') == 7)
    mkdir(AssoCode.savePath);
end
% dirNow = cd(fullfile(AssoCode.Path_RootDir, 'BatchDataSet'));
% if ~(exist(AssoCode.parFoldName, 'dir') == 7)
%     mkdir(fullfile(AssoCode.Path_RootDir, 'BatchDataSet'));
% end

str = datestr(now, 'yymmddHHMM');
AssoCode.Layer2.saveName = [AssoCode.Layer2.saveName, '_', str(1:6), ...
    '_', str(7:end)];

save(fullfile(AssoCode.savePath, [AssoCode.Layer2.saveName, '.mat']), 'AssoCode');

% cd(dirNow);

fprintf('Done!\n')
fprintf('=================================================================\n')