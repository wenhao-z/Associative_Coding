function [nlOutput, nlFunc_Edge] = nonlinearFunc(filterImgPatch, Layer1)
% The nonlinearity after Gabor's output
% filterImgPatch: [num. of channel, num of images]
% Layer1: a struct of AssoCode storing the parameters of preprocessing
%         units, including Gabor filters and nonlienarities

% Author: Wen-Hao Zhang, June-9, 2017
% wenhaoz1@andrew.cmu.edu
% @Carnegie Mellon University


switch Layer1.flagNonLincFunc
    case 0
        % A rectified linear function
        fprintf('Nonlinearity: a negative rectified function.\n')
        nlOutput = filterImgPatch;
        nlOutput(nlOutput<0) = 0;
        
        nlFunc_Edge = [];
    case 1
        % Cumulative distribution of ABSOLUTE value of Gabor's output
        fprintf('Nonlinearity: cdf. of the ABSOLUTE value of all filters output (Max Entropy).\n')       
        filterImgPatch = abs(filterImgPatch);
        
        if ~isfield(Layer1, 'nlFunc_Edge')
            IntBins = floor(numel(filterImgPatch)/ 2^Layer1.depthNonlienarity);
            Edge = sort(filterImgPatch(:));
            nlFunc_Edge = [Edge(1: IntBins: end)', Inf];
        else
            nlFunc_Edge = Layer1.nlFunc_Edge;
        end
        
        [~, ~, bins] = histcounts(filterImgPatch, nlFunc_Edge);
        nlOutput = Layer1.meanFiringRate * bins/ length(nlFunc_Edge);
    case 2
        % Cumulative distribution
        fprintf('Nonlinearity: cdf of all filters outputs (Max Entropy).\n')
        if ~isfield(Layer1, 'nlFunc_Edge')
            IntBins = floor(numel(filterImgPatch)/ 2^Layer1.depthNonlienarity);
            Edge = sort(filterImgPatch(:));
            nlFunc_Edge = [Edge(1: IntBins: end)', Inf];
        else
            nlFunc_Edge = Layer1.nlFunc_Edge;
        end
        
        [~, ~, bins] = histcounts(filterImgPatch, nlFunc_Edge);
        nlOutput = Layer1.meanFiringRate * bins/ length(nlFunc_Edge);
    case 3
             % Cumulative distribution of RECTIFIED value of Gabor's output
        fprintf('Nonlinearity: cdf. of the RECTIFIED value of all filters output (Max Entropy).\n')       
        filterImgPatch(filterImgPatch<0) = 0;
        
        if ~isfield(Layer1, 'nlFunc_Edge')
            IntBins = floor(numel(filterImgPatch)/ 2^Layer1.depthNonlienarity);
            Edge = sort(filterImgPatch(:));
            nlFunc_Edge = [Edge(1: IntBins: end)', Inf];
        else
            nlFunc_Edge = Layer1.nlFunc_Edge;
        end
        
        [~, ~, bins] = histcounts(filterImgPatch, nlFunc_Edge);
        nlOutput = Layer1.meanFiringRate * bins/ length(nlFunc_Edge);
end
