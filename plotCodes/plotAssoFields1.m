function hFig = plotAssoFields1(BMStruct, hFig)
% Plot associative fields of the neuron prefering 0 deg at center
% hypercolumn

% Wen-Hao Zhang, Oct-1, 2016
% @Carnegie Mellon University

% Data structure
% BM.W22 [Height, Width, # of Gabor]

%% Plot connections among hypercolumns

if exist('hFig', 'var')
    figure(hFig);
else
    hFig = figure(1);
end
hold on
axis square

szHiddenNeuron = [BMStruct.numHyperCol, BMStruct.numSpatFreq, BMStruct.numOrient];

% Get the index of neurons
IdxNeuron = 1: prod(szHiddenNeuron);
IdxNeuron = reshape(IdxNeuron, szHiddenNeuron);
IdxNeuron = permute(IdxNeuron, [3,4, 1, 2]); % [spatial frequency, orientation, height, width]

szIdxNeuron = size(IdxNeuron);

% The index of the center neuron 0 deg neuron in center hypercolumn
IdxCenterNeuron = IdxNeuron(1, BMStruct.numOrient/2, ...
    (BMStruct.numHyperCol(1)+1)/2, (BMStruct.numHyperCol(2)+1)/2);

% Get the plotted weight array
WPlot = BMStruct.W22(IdxCenterNeuron, IdxNeuron(:,:, :, :));
WPlot = reshape(WPlot, szIdxNeuron); % [spatial frequency, orientation, height, width]

WPlot(:,:, (BMStruct.numHyperCol(1)+1)/2, (BMStruct.numHyperCol(2)+1)/2) = nan;

% szIdxNeuron(3) = szIdxNeuron(3) - 1;
% szIdxNeuron(4) = szIdxNeuron(4) - 1;
% WPlot

% Set the color of line
cMin = min(WPlot(:));
cMax = max(WPlot(:));
cmLength = 128;
cMap = jet(cmLength);
colormap(cMap);

% cMapPos = spring(cmLength);
% cMapNeg = cool(cmLength);
% cMaxPos = max(WPlot(:));
% cMinPos = min(WPlot(:));

lenBar = 0.3; % length of bar

% Plot the preferred bar of example neuron
plot(lenBar * [-1, 1], zeros(1,2), 'k', 'linew', 1.5)


for IdxDim3 = 1: szIdxNeuron(3) % height
    for IdxDim4 = 1: szIdxNeuron(4) % width
        W_HC = WPlot(:,:, IdxDim3, IdxDim4); % W at the same Hyper-column (HC)
        
        [~, IdxOrient(1)] = max(W_HC);
        [~, IdxOrient(2)] = min(W_HC);
        
        for iterOrient = IdxOrient
            [X, Y] = pol2cart(BMStruct.OrientArray(iterOrient), lenBar);
            X = X * [-1, 1];
            Y = Y * [-1, 1]; % the location of two ends of a bar
            X = X + (IdxDim4 - (szIdxNeuron(4)+1)/2); % plus the location of hypercolumn
            Y = Y + (IdxDim3 - (szIdxNeuron(3)+1)/2);
            
            cSpec = fix((WPlot(1, iterOrient, IdxDim3, IdxDim4)-cMin)/(cMax-cMin)*cmLength)+1;
            cSpec = cMap(min(cSpec, cmLength), :);
            
            lineWidth = 20 * abs(WPlot(1, iterOrient, IdxDim3, IdxDim4));
            
            if WPlot(1, iterOrient, IdxDim3, IdxDim4) > 0
                plot(X, Y, 'linewidth', lineWidth, 'color', cSpec);
            elseif WPlot(1, iterOrient, IdxDim3, IdxDim4) < 0
                plot(X, Y, '-.', 'linewidth', lineWidth, 'color', cSpec);
            end
        end
    end
end

colorbar('Ticks', [0, -cMin/(cMax-cMin), 1],...
         'TickLabels', [cMin, 0, cMax])