function hFig = plotBMWeight(BMStruct, hFig)
% Plot the trained weights in a boltzmann machine
% This code is used inside mfFitBM.m
% Wen-Hao Zhang, Sep-13, 2016
% @Carnegie Mellon University

% Data structure
% BM.W22 [Height, Width, # of Gabor]


% Plot the weights within the same hypercolumn

%% Plot connections among hypercolumns

if exist('hFig', 'var')
    figure(hFig);
else
    hFig = figure(1);
end

set(hFig, 'position', [0, 1600, 1400, 700]);
% set(hFig, 'position', [0, 800, 800, 1000]);

% hAxe(1) = subplot(1,2,1); axis square
hAxe(1) = subplot(1,3,1); axis square

szHiddenNeuron = [BMStruct.numHyperCol, BMStruct.numSpatFreq, BMStruct.numOrient];

% Get the index of neurons
IdxNeuron = 1: prod(szHiddenNeuron);
IdxNeuron = reshape(IdxNeuron, szHiddenNeuron);
IdxNeuron = permute(IdxNeuron, [3,4, 1, 2]); % [spatial frequency, orientation, height, width]

numNeuronPerHyperCol = BMStruct.numSpatFreq*BMStruct.numOrient;

WPlot = BMStruct.W22(IdxNeuron(:), IdxNeuron(:));
transIdx = numNeuronPerHyperCol * ones(1, prod(BMStruct.numHyperCol));
WPlot = mat2cell(WPlot, transIdx, transIdx);

maxWeight = max(BMStruct.W22(:));
% add edge between connnection patches
WPlot = cellfun(@(x) [[x, maxWeight * ones(numNeuronPerHyperCol, 1)]; ...
    maxWeight * ones(1, numNeuronPerHyperCol+1)], WPlot, 'uniformout', 0);
WPlot = cell2mat(WPlot);
WPlot = WPlot(1:end-1, 1:end-1);

% Plot
imagesc(WPlot); 
axis square
hold on
plot([1, length(WPlot)], [1, length(WPlot)], 'w', 'linew', 1.5)
colormap gray
colorbar('location', 'eastoutside')

% Labeling plot axis
yTickLoc = ((1: prod(BMStruct.numHyperCol)) - 1) * (numNeuronPerHyperCol+1) ...
    + numNeuronPerHyperCol/2;

yTickLab = 1: prod(BMStruct.numHyperCol);
[rSub, cSub] = ind2sub(BMStruct.numHyperCol, yTickLab);
yTickLab = [num2str(rSub'), repmat(',', length(yTickLab),1), num2str(cSub')];
yTickLab = [repmat('(', length(yTickLab), 1), yTickLab, repmat(')', length(yTickLab),1)];

set(gca, 'ytick', yTickLoc, 'yticklabel', yTickLab, ...
    'xtick', [1, 16], 'xticklabel', ...
    BMStruct.OrientArray(end)*180/pi*[-1, 1]);
ylabel('Location of hypercolumns')

%%
% Plot the weights from the 0 deg neuron in center hypercolumn with all
% neurons from hypercolummn at the same row

% The index of the center neuron 0 deg neuron in center hypercolumn
IdxCenterNeuron = IdxNeuron(1, BMStruct.numOrient/2, ...
    (BMStruct.numHyperCol(1)+1)/2, (BMStruct.numHyperCol(2)+1)/2);

% hAxe(2) = subplot(1,4,3); 
hAxe(2) = subplot(1,3,2); 

szIdxNeuron = size(IdxNeuron);

WPlot = BMStruct.W22(IdxCenterNeuron, IdxNeuron(:,:, (BMStruct.numHyperCol(1)+1)/2, :));
WPlot = squeeze(reshape(WPlot, szIdxNeuron([1,2,4])));

% x and y values for plotting
xVal = (1: BMStruct.numHyperCol(2)) - (1+BMStruct.numHyperCol(2))/2;
yVal = (BMStruct.OrientArray - BMStruct.OrientArray(BMStruct.numOrient/2))*180/pi;
contourf(xVal, yVal, WPlot);

pbaspect([1, 2.5, 1])
% axis square
% colorbar('location', 'southoutside')

ylabel('Orientation')
xlabel('Horizontal Location')
title({'Neuron 0^\circ@(3,3)', 'to others at the same row'})
set(gca, 'xtick', xVal, 'ytick', [yVal(1), yVal(end/2), yVal(end)])

colormap(hAxe(2), parula)
caxis([min(BMStruct.W22(:)), max(BMStruct.W22(:))])

%%
% Plot the weights from the 0 deg neuron in center hypercolumn with all
% neurons from hypercolummn at the same column
% hAxe(3) = subplot(1,4,4);
hAxe(3) = subplot(1,3,3);

WPlot = BMStruct.W22(IdxCenterNeuron, IdxNeuron(:,:, :, (BMStruct.numHyperCol(1)+1)/2));
WPlot = squeeze(reshape(WPlot, szIdxNeuron(1:3)));

% x and y values for plotting
xVal = (1: BMStruct.numHyperCol(1)) - (1+BMStruct.numHyperCol(1))/2;
yVal = (BMStruct.OrientArray - BMStruct.OrientArray(BMStruct.numOrient/2))*180/pi;

% imagesc(xVal, yVal, WPlot');
contourf(xVal, yVal, WPlot);
pbaspect([1, 2.5, 1])
colorbar('location', 'eastoutside')

xlabel('Vertical Location')
ylabel('Orientation')
% ylabel('Orientation')
title({'Neuron 0^\circ@(3,3)', 'to others at the same column'})
set(gca, 'xtick', xVal, 'ytick', [yVal(1), yVal(end/2), yVal(end)])

colormap(hAxe(3), parula)
caxis([min(BMStruct.W22(:)), max(BMStruct.W22(:))])

% [left bottom width height]
set(hAxe(1), 'position', [0.05, 0.15, 0.5, 0.75])
set(hAxe(2), 'position', [0.55, 0.18, 0.2, 0.72])
set(hAxe(3), 'position', [0.75, 0.18, 0.2, 0.72])
 
% set(hAxe(1), 'position', [0.05, 0.1, 0.45, 0.9])
% set(hAxe(2), 'position', [0.6, 0.1, 0.13, 0.8])
% set(hAxe(3), 'position', [0.8, 0.1, 0.13, 0.8])
