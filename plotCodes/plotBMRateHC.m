function hFig = plotBMRateHC(Rate, BMStruct, hFig)
% Plot the neuronal activities of every hyper-column

% INPUT:
% Rate: [N, 2]

% Wen-Hao Zhang, Oct-11, 2016
% wenhaoz1@andrew.cmu.edu
% @Carnegie Mellon University

if exist('hFig', 'var')
    figure(hFig);
else
    hFig = figure(1);
end

% Dim of szHiddenNeuron [height, width, spatial frequency, orientation]
szHiddenNeuron = [BMStruct.numHyperCol, BMStruct.numSpatFreq, BMStruct.numOrient];

Rate = reshape(Rate, [szHiddenNeuron, size(Rate, 2)]);

yLim = [min(Rate(:)), max(Rate(:))];
if yLim(1) == yLim(2)
   yLim(1) = yLim(1) - 1; 
   yLim(2) = yLim(2) + 1;
end
% cSpec = lines(64);
for iter = 1: prod(BMStruct.numHyperCol)
    subplot(BMStruct.numHyperCol(1), BMStruct.numHyperCol(2), iter);
    [Idx1, Idx2] = ind2sub(BMStruct.numHyperCol, iter);
    plot([-BMStruct.OrientArray(end), BMStruct.OrientArray]*180/pi, squeeze(Rate(Idx1, Idx2,:, [end, 1:end], :)))
    hold on
    set(gca, 'xtick', [-BMStruct.OrientArray(end), 0, BMStruct.OrientArray(end) ] *180/pi)
    axis tight; axis square
    ylim(yLim);
end

Idx = sub2ind(BMStruct.numHyperCol, (BMStruct.numHyperCol(2)+1)/2, BMStruct.numHyperCol(1));
subplot(BMStruct.numHyperCol(1), BMStruct.numHyperCol(2), Idx);
xlabel('Orientation')

% hPlot(1) = plot(0,0, 'color', cSpec(1,:));
% hPlot(2) = plot(0,0, 'color', cSpec(2,:));
% legend(hPlot, 'r_2 (r_1 = 0)', 'b_2', 'location', 'best')
