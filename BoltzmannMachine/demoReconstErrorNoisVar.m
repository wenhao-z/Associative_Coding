% Demonstrate how reconstruction error is dependent on noise strength

% Wen-Hao Zhang, Oct-11, 2016
% wenhaoz1@andrew.cmu.edu
% @Carnegie Mellon University

% Set the working path
setWorkPath;

% Load learnt Boltzmann machine
fileName = fullfile(Path_RootDir, 'BatchDataSet', 'HC9_RF9', 'trainBMPars(28-Sep).mat');
load(fileName);

BMStruct = AssoCode.Layer2;
BMStruct.maxIter = 1e4;
BMStruct.tolDX = 1e-15;
BMStruct.coefIter = 0.1;
nRandSearch = 5;

% Load one InputBM
% fileName = fullfile(Path_RootDir, 'BatchDataSet', 'HC9_RF9', 'TrainDatSet.mat');
fileName = fullfile(Path_RootDir, 'BatchDataSet', 'HC9_RF9', 'InputBM.mat');
load(fileName, 'InputBM')
InputBM = InputBM(:,1);

%%
stdNois = [0, logspace(-4, 1, 10)];
nTrial = 1e2;

r2ss = zeros(BMStruct.numHiddenNeuron, nTrial, length(stdNois));
r1ss = zeros(BMStruct.numHiddenNeuron, nTrial, length(stdNois));
Input = zeros(BMStruct.numHiddenNeuron, nTrial, length(stdNois));

for iter = 1: length(stdNois)
    fprintf('Progress: %d/%d.\n', iter, length(stdNois));
    for iterTrial = 1: nTrial
%         fprintf('Trial NO.: %d.', iterTrial)
        % Feedforward pass
        Input(:, iterTrial, iter) = InputBM + stdNois(iter) * randn(size(InputBM) );
        Iff = bsxfun(@times, BMStruct.W12', Input(:, iterTrial, iter));
        Iff = bsxfun(@plus, Iff, BMStruct.Bias2);
        
        r2init = 1./(1 + exp(-Iff)); % use the activation when only receiving feedforward inputs as initialization
        r2ss(:,iterTrial, iter) = getBMEqubrmState(r2init, BMStruct.W22, Iff, BMStruct);
        
        % Feedback pass
        % Feedback inputs from layer 2 (hidden layer) to layer 1 (visible layer)
        Ifb = bsxfun(@times, BMStruct.W12', r2ss(:,iterTrial, iter));
        Ifb = bsxfun(@plus, Ifb, BMStruct.Bias1);
        
        r1init = 1./(1 + exp(-Ifb));
        r1ss(:,iterTrial, iter) = getBMEqubrmState(r1init, BMStruct.W11, Ifb, BMStruct);
    end
end

%% plot
Error_NoisInput = r1ss - Input;
Error_OrigInput = bsxfun(@minus, r1ss, InputBM);

Error_NoisInput = squeeze(mean(mean(Error_NoisInput.^2, 1), 2));
Error_OrigInput = squeeze(mean(mean(Error_OrigInput.^2, 1), 2));

% dr1ss = bsxfun(@minus, r1ss, r1ss(:,1,1));
% loglog(stdNois, squeeze(mean(var(dr1ss, [], 2), 1)))
% loglog(stdNois, Error_NoisInput)
% hold on
% loglog(stdNois, Error_OrigInput)

subplot(1,2,1)
[hAxe, hLine(1), hLine(2)] = plotyy(stdNois, Error_NoisInput, stdNois, Error_OrigInput);
set(hAxe, 'xscale', 'log', 'yscale', 'log', 'YTickMode', 'auto');
set(hLine, 'linew', 1.5)
axes(hAxe(1)); axis square
axes(hAxe(2)); axis square

xlabel('Std. of noise')
ylabel(hAxe(1), 'Reconstrction error with noisy input')
ylabel(hAxe(2), 'Reconstrction error with original input')

% legend('Reconstruction error with noisy input', ...
%     'Reconstruction error with original input', 'location', 'best')

subplot(1,2,2)
% semilogx(stdNois, 1 - Error_OrigInput'./ stdNois.^2, 'linew', 1.5)
semilogx(stdNois, 1-Error_OrigInput./ Error_NoisInput, 'linew', 1.5)
axis square