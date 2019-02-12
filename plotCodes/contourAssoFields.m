function hFig = contourAssoFields(BMStruct, hFig)
% Plot the trained weights in a boltzmann machine
% This code is used inside mfFitBM.m
% Wen-Hao Zhang, Sep-13, 2016
% @Carnegie Mellon University

% Data structure
% BM.W22 [Height, Width, # of Gabor]


% Plot the weights within the same hypercolumn


% Index of neurons
szHiddenNeuron = [BMStruct.numHyperCol, BMStruct.numSpatFreq, BMStruct.numOrient];

% Get the index of neurons
IdxNeuron = 1: prod(szHiddenNeuron);
IdxNeuron = reshape(IdxNeuron, szHiddenNeuron);
IdxNeuron = permute(IdxNeuron, [3,4, 1, 2]); % [spatial frequency, orientation, height, width]

szIdxNeuron = size(IdxNeuron);
IdxNeuronCenterHC = IdxNeuron(:,:, (szIdxNeuron(3)+1)/2, (szIdxNeuron(4)+1)/2);

numNeuronPerHyperCol = BMStruct.numSpatFreq*BMStruct.numOrient;

%% Plot connections within the center hypercolumn

if exist('hFig', 'var')
    figure(hFig);
else
    hFig = figure(1);
end

set(hFig, 'position', [0, 1600, 1400, 700]);
% set(hFig, 'position', [0, 800, 800, 1000]);

% hAxe(1) = subplot(1,2,1); axis square
hAxe(1) = subplot(1,3,1); axis square

WPlot = BMStruct.W22(IdxNeuronCenterHC(:), IdxNeuronCenterHC(:));
WPlot = reshape(WPlot, numNeuronPerHyperCol, numNeuronPerHyperCol);

% Plot
imagesc(WPlot); 
axis square
% colormap gray
colorbar('location', 'eastoutside')

set(gca, 'xtick', [1, 16], 'xticklabel', ...
    BMStruct.OrientArray(end)*180/pi*[-1, 1], ...
    'ytick', [1, 16], 'yticklabel', ...
    BMStruct.OrientArray(end)*180/pi*[-1, 1]);
xlabel('Orientation')
ylabel('Orientation')

%% Average connections within all hypercolumns
WAvg_IntraHC = zeros(numNeuronPerHyperCol, numNeuronPerHyperCol, szIdxNeuron(3));
for iter = 1: szIdxNeuron(3)
    Idx = IdxNeuron(:,:, iter, iter);
    WAvg_IntraHC(:,:,iter) = BMStruct.W22(Idx(:), Idx(:));
end
WAvg_IntraHC = WAvg_IntraHC / szIdxNeuron(3);


%% Get the associative fields

% The index of the center neuron 0 deg neuron in center hypercolumn
IdxCenterNeuron = IdxNeuron(1, BMStruct.numOrient/2, ...
    (BMStruct.numHyperCol(1)+1)/2, (BMStruct.numHyperCol(2)+1)/2);

% Horizontal direction
WPlot1 = BMStruct.W22(IdxCenterNeuron, IdxNeuron(:,:, (BMStruct.numHyperCol(1)+1)/2, :));
WPlot1 = squeeze(reshape(WPlot1, szIdxNeuron([1,2,4])));

% Vertical direction
WPlot2 = BMStruct.W22(IdxCenterNeuron, IdxNeuron(:,:, :, (BMStruct.numHyperCol(1)+1)/2));
WPlot2 = squeeze(reshape(WPlot2, szIdxNeuron(1:3)));

WPlot = cat(3, WPlot1, WPlot2);
WPlot(:, (end+1)/2, :) = [];
WPlot = cat(1, WPlot(end,:,:), WPlot);

clear WPlot1 WPlot2

%%
% Plot the weights from the 0 deg neuron in center hypercolumn with all
% neurons from hypercolummn at the same row

% hAxe(2) = subplot(1,4,3); 
hAxe(2) = subplot(1,3,2); 

% x and y values for plotting
xVal = (1: BMStruct.numHyperCol(2)) - (1+BMStruct.numHyperCol(2))/2;
yVal = (BMStruct.OrientArray - BMStruct.OrientArray(BMStruct.numOrient/2))*180/pi;
yVal = [-yVal(end), yVal];
xVal((end+1)/2) = [];

% imagesc(xVal, yVal, WPlot(:,:,1));
contourf(xVal, yVal, WPlot(:,:,1));

pbaspect([1, 2.5, 1])
% axis square
% colorbar('location', 'southoutside')

ylabel('Orientation')
xlabel('Horizontal Location')

title({['Neuron 0^\circ@(', num2str((BMStruct.numHyperCol(1)+1)/2) ',' num2str((BMStruct.numHyperCol(1)+1)/2) ')'], 'to others at the same row'})
set(gca, 'xtick', xVal, 'ytick', [yVal(1), yVal((end+1)/2), yVal(end)])

colormap(hAxe(2), parula)
caxis([min(WPlot(:)), max(WPlot(:))])

%%
% Plot the weights from the 0 deg neuron in center hypercolumn with all
% neurons from hypercolummn at the same column
% hAxe(3) = subplot(1,4,4);
hAxe(3) = subplot(1,3,3);


% x and y values for plotting
xVal = (1: BMStruct.numHyperCol(1)) - (1+BMStruct.numHyperCol(1))/2;
yVal = (BMStruct.OrientArray - BMStruct.OrientArray(BMStruct.numOrient/2))*180/pi;
yVal = [-yVal(end), yVal];
xVal((end+1)/2) = [];

% imagesc(xVal, yVal, WPlot(:,:,2));
contourf(xVal, yVal, WPlot(:,:,2));
pbaspect([1, 2.5, 1])
colorbar('location', 'eastoutside')

xlabel('Vertical Location')
ylabel('Orientation')
% ylabel('Orientation')
title({['Neuron 0^\circ@(', num2str((BMStruct.numHyperCol(1)+1)/2) ',' num2str((BMStruct.numHyperCol(1)+1)/2) ')'], 'to others at the same column'})
set(gca, 'xtick', xVal, 'ytick', [yVal(1), yVal((end+1)/2), yVal(end)])

colormap(hAxe(3), parula)
caxis([min(WPlot(:)), max(WPlot(:))])

% [left bottom width height]
set(hAxe(1), 'position', [0.05, 0.15, 0.5, 0.75])
set(hAxe(2), 'position', [0.55, 0.18, 0.2, 0.72])
set(hAxe(3), 'position', [0.75, 0.18, 0.2, 0.72])
 
% set(hAxe(1), 'position', [0.05, 0.1, 0.45, 0.9])
% set(hAxe(2), 'position', [0.6, 0.1, 0.13, 0.8])
% set(hAxe(3), 'position', [0.8, 0.1, 0.13, 0.8])
