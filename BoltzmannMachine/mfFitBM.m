function AssoCode = mfFitBM(InputBM, AssoCode)
% Fitting a Boltzmann machine through mean field approximation
% Details refer to
% Welling et al., A New Learning Algorithm for Mean Field Boltzmann Machines

% InputBM: all training data. [m x number of samples] array.
% BMSturct: a struct storing the parameters of a Boltzmann machine and optimization.
%
% Outputs
% BMStruct: updating its parameters

% Wen-Hao Zhang
% @Carnegie Mellon Universiy, Sep-9, 2016.

%% Extract some commonly used variables from BMStruct
BMStruct = AssoCode.Layer2;
if ~isfield(BMStruct, 'iterEpoch')
    BMStruct.iterEpoch = 1;
end

% Re-initialize Bias1 (visible layer) according to the mean value of inputs
Bias1 = mean(InputBM, 2);
BMStruct.Bias1 = log(Bias1./ (1 - Bias1));
clear Bias1

% Initialize a struct storing input/output and gradients for updating
% parameters
NetActv = struct(...
    'dW22', zeros(size(BMStruct.W22)), ...
    'dW12', zeros(size(BMStruct.W12)),...
    'dW11', zeros(size(BMStruct.W11)), ...
    'dBias1', zeros(size(BMStruct.Bias1)), ...
    'dBias2', zeros(size(BMStruct.Bias2)), ...
    'r1Pos', [], ...
    'r2Pos', [], ...
    'r1Neg', [], ...
    'r2Neg', []);

%% Initialize the variables monitoring learning process
if ~isfield(BMStruct, 'ReConstError')
    ReConstError = zeros(1, AssoCode.Layer2.maxEpoch);
else
    ReConstError = BMStruct.ReConstError;
end

if BMStruct.bGPU
    % Copy variables from MEMORY into GPU
    ReConstError = gpuArray(single(ReConstError));
    InputBM = gpuArray(single(InputBM));
    for varName = fieldnames(NetActv)'
        NetActv.(varName{1}) = gpuArray(single(NetActv.(varName{1})));
    end
    for varName = fieldnames(BMStruct)'
        if ~ischar(BMStruct.(varName{1}))
            BMStruct.(varName{1}) = gpuArray(single(BMStruct.(varName{1})));
        end
    end
    clear varName
end

if BMStruct.bPlotinLearn
    close all;
    for iter = 1: 2
        hFig(iter) = figure;
    end
    clear iter
end

%%
fprintf('Start training a Boltzmann machine.\n')
for iterEpoch = BMStruct.iterEpoch: BMStruct.maxEpoch
    fprintf('%d/%d epoches.  ', iterEpoch, BMStruct.maxEpoch);
    tic
    
    % Shuffle the order of traning examples every epoch
    IdxRandPerm = randperm(size(InputBM,2));
    
    for iterBatch = 1: BMStruct.numBatchPerEpoch-1
        %% Get the batch training set
        Idx = [(iterBatch-1)*BMStruct.szBatch + 1, ...
            min(iterBatch * BMStruct.szBatch, size(InputBM, 2))];
        
        NetActv.r1Pos = InputBM(:, IdxRandPerm(Idx(1): Idx(2)));
        NetActv.szBatch = diff(Idx) + 1; % number of training examples
        
        %% Contrastive divergence mean field learning
        % Calulate the mean activation of visible and hidden units through
        % contrastive divergence
        NetActv = mfContrastDivg(BMStruct, NetActv);
        ReConstError(iterEpoch) = (1 - 1./iterBatch) * ReConstError(iterEpoch) ...
            + mean(sqrt((NetActv.r1Pos(:) - NetActv.r1Neg(:)).^2))/iterBatch;

        %% Updating the parameters of Boltzmann machine
        [BMStruct, NetActv] = updateBMPars(BMStruct, NetActv);
    end
    fprintf('ReconsError: %d.\n', ReConstError(iterEpoch))
    fprintf('Bias1: min: %d; max: %d; median: %d.\n', min(BMStruct.Bias1(:)), ...
        median(BMStruct.Bias1(:)), max(BMStruct.Bias1(:)));
    fprintf('Bias2: min: %d; max: %d; median: %d.\n', min(BMStruct.Bias2(:)), ...
        median(BMStruct.Bias2(:)), max(BMStruct.Bias2(:)));
    fprintf('W22: min: %d; max: %d; median: %d.\n', min(BMStruct.W22(:)), ...
        median(BMStruct.W22(:)), max(BMStruct.W22(:)));
    toc
    fprintf('------------------------------------------------\n')

    %% Fold updated parameters into BMStruct and SAVE after each epoch
    if (~mod(iterEpoch, BMStruct.nEpochtoSave)) || (iterEpoch ==1)
        BMStruct.iterEpoch = iterEpoch;
        BMStruct.ReConstError = ReConstError;
       
        if BMStruct.bGPU
            % Copy variables from GPU to MEMORY
            for varName = fieldnames(BMStruct)'
                if ~ischar(BMStruct.(varName{1}))
                    AssoCode.Layer2.(varName{1}) = double(gather(BMStruct.(varName{1})));
                end
            end
        else
            AssoCode.Layer2 = BMStruct;
        end        
               
        
        % Save updated BMStruct
        fileName = [BMStruct.saveName, '.mat'];
        fileName = fullfile(AssoCode.savePath, fileName);
        save(fileName, 'AssoCode','-append');
        
        % Plot intermediate results
        %         if BMStruct.bPlotinLearn
        %             plotBMWeight(BMStruct, hFig(1));
        %             plotMonitorLearning(BMStruct, NetActv, hFig(2));
        %             drawnow;
        %         end
    end
    
end
