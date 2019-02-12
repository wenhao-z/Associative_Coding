% Search attractors in mean field Boltzmann machine through random
% initialization
% Wen-Hao Zhang, Oct-11, 2016
% wenhaoz1@andrew.cmu.edu
% @Carnegie Mellon University 

% Set the working path
setWorkPath;

fileName = fullfile(Path_RootDir, 'BatchDataSet', 'HC9_RF9', 'trainBMPars(28-Sep).mat');
load(fileName);

BMStruct = AssoCode.Layer2;
BMStruct.maxIter = 1e4;
BMStruct.tolDX = 1e-15;
BMStruct.coefIter = 0.05;
nRandSearch = 5;

% Load one InputBM
% fileName = fullfile(Path_RootDir, 'BatchDataSet', 'HC9_RF9', 'TrainDatSet.mat');
fileName = fullfile(Path_RootDir, 'BatchDataSet', 'HC9_RF9', 'InputBM.mat');
load(fileName, 'InputBM')
InputBM = InputBM(:,1);

%% Search self-sustained attractors (without feedforward input)
Input = [InputBM, zeros(BMStruct.numVisNeuron, 1)];
Iff = bsxfun(@times, BMStruct.W12', Input);
Iff = bsxfun(@plus, Iff, BMStruct.Bias2);

r2ss0 = getBMEqubrmState(zeros(size(Iff)), BMStruct.W22, Iff, BMStruct);
r2init = zeros([size(Iff), nRandSearch]);
r2ss = zeros([size(Iff), nRandSearch]);

for iter = 1: nRandSearch
    fprintf('Progress: %d/%d.\n', iter, nRandSearch)
    %     r2init(:,iter) = 100*ones(BMStruct.numHiddenNeuron, 1); % random initialization
    r2init(:, :, iter) = r2ss0 + 10*randn(size(Iff)); % random initialization
    % r2init = zeros(BMStruct.numHiddenNeuron, 1); % zero initialization
    
    r2ss(:,:, iter) = getBMEqubrmState(r2init(:,:, iter), BMStruct.W22, Iff, BMStruct);
end

%% Linear stability analysis around a fixed point

JacobMat = getJacobMatrix(BMStruct, r2ss(:,1,1));
[eigVector, eigValue] = eig(JacobMat);

[eigValue, Idx] = sort(diag(eigValue), 'descend');
eigVector = eigVector(:, Idx);
clear Idx
%%
% figure
% plot(BMStruct.Bias2)
% hold on
% plot(squeeze(r2ss(:,1,:)))
% plot(squeeze(r2ss(:,2,:)))

plotBMRateHC([r2ss(:,1,1), InputBM], BMStruct);
