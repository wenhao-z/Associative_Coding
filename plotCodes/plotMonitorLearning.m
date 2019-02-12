function hFig = plotMonitorLearning(BMStruct, NetActv, hFig)
% Plot variables which monitor the learning process of a Boltzmann machine
% This code is used inside mfFitBM.m
% Wen-Hao Zhang, Sep-21, 2016
% @Carnegie Mellon University

% Data structure

if exist('hFig', 'var')
    figure(hFig);
else
    hFig = figure(2);
end

set(hFig, 'position', [900, 800, 600, 700]);
% [left bottom width height]
%% Histogram of weights in hidden layer
subplot(3,2,1);
hW22 = triu(BMStruct.W22);
hW22(hW22==0) = [];
[hW22, WEdge] = histcounts(hW22, 1e3);
% hW22 = histogram(hW22, 1e3);
hW22 = hW22 / length(hW22);
plot(WEdge(1:end-1), hW22);
xlabel('W_{22}')
ylabel('Histogram')

%% Histogram of change of weights in hidden layer
subplot(3,2,2)
hW22 = triu(NetActv.dW22);
hW22(hW22==0) = [];
[hW22, WEdge] = histcounts(hW22, 1e3);
% hW22 = histogram(hW22, 1e3);
hW22 = hW22 / length(hW22);
plot(WEdge(1:end-1), hW22);
xlabel('\DeltaW_{22}')
ylabel('Histogram')

%% Histogram of biases
subplot(3,2,3)
Bias = [BMStruct.Bias1, BMStruct.Bias2];
cSpec = lines(2);

for iter = 1: 2
    [hBias, BEdge] = histcounts(Bias(:, iter), 1e2);
    hBias = hBias/ length(hBias);
    plot(BEdge(1:end-1), hBias, 'color', cSpec(iter,:));
    hold on
end
hold off
legend('Bias 1', 'Bias 2', 'location', 'best', 'orientation', 'horizontal')
xlabel('Bias')
ylabel('Histogram')


%% Histogram of changes of biases
subplot(3,2,4)
Bias = [NetActv.dBias1, NetActv.dBias2];
cSpec = lines(2);

for iter = 1: 2
    [hBias, BEdge] = histcounts(Bias(:, iter), 1e2);
    hBias = hBias/ length(hBias);
    plot(BEdge(1:end-1), hBias, 'color', cSpec(iter,:));
    hold on
end
hold off
legend('\DeltaBias 1', '\DeltaBias 2', 'location', 'best', 'orientation', 'horizontal')
xlabel('\DeltaBias')
ylabel('Histogram')

%% Probability of activation of hidden units
subplot(3,2,5)
imagesc(NetActv.r2Pos);
xlabel('Training examples')
ylabel('Neurons')
colormap gray
caxis([0, 1]);
% colorbar
title({'Firing Prob. of', 'Hidden Neurons'})

%% Reconstruction error
subplot(3,2,6);
plot(BMStruct.ReConstError(1: BMStruct.iterEpoch));
ylabel('Reconstruction error');


%% Monitor overfitting