function [BMStruct, NetActv] = updateBMPars(BMStruct, NetActv)
% Given the activities of neurons in Boltzmann machine
% in positive (clamped with data) and negative (free evolution) phase,
% updating the network parameters.

% Wen-Hao Zhang, Sep-22, 2016
% @Carnegie Mellon University


% Unfold some common parameters from BMStruct
learnRate = BMStruct.learnRate;
lambdaMomentum = BMStruct.lambdaMomentum;
lambdaL2RegW22 = BMStruct.lambdaL2RegW22;

%% Gradient of W22 (lateral connections in hidden layer)
if BMStruct.bLateralW22
    dW22 = (NetActv.r2Pos * NetActv.r2Pos' - NetActv.r2Neg * NetActv.r2Neg') / NetActv.szBatch;
else
    dW22 = mean(NetActv.r2Pos.^2,2) - mean(NetActv.r2Neg.^2, 2);
    dW22 = diag(dW22);
end

dW22 = dW22 - lambdaL2RegW22 * BMStruct.W22;
dW22 = dW22 - diag(diag(dW22)); % Diagonal terms are zero
% Make the gradient array are STRICTLY symmetric to avoid numerical error
dW22 = (dW22 + dW22')/2; 

NetActv.dW22 = learnRate * dW22 + lambdaMomentum * NetActv.dW22;

%% Gradient of W11 (lateral connections in visible layer)
if BMStruct.bLearnW11
    if BMStruct.bLateralW22
        dW11 = (NetActv.r1Pos * NetActv.r1Pos' - NetActv.r1Neg * NetActv.r1Neg') / NetActv.szBatch;
    else
        dW11 = mean(NetActv.r1Pos.^2,2) - mean(NetActv.r1Neg.^2, 2);
        dW11 = diag(dW11);
    end
    
    dW11 = (dW11 + dW11')/2;
    NetActv.dW11 = learnRate * dW11 + lambdaMomentum * NetActv.dW11;
end

%% Gradient of W12 (connections between hidden and visible layer)
if BMStruct.bLearnW12
    %     dW12 = (NetActv.r1Pos * NetActv.r2Pos' - NetActv.r1Neg * NetActv.r2Neg')/ NetActv.szBatch;
    
    % Visible and hidden neurons are one-to-one connected.
    dW12 = NetActv.r1Pos .* NetActv.r2Pos - NetActv.r1Neg .* NetActv.r2Neg;
    dW12 = mean(dW12, 2);
    NetActv.dW12 = learnRate * dW12' + lambdaMomentum * NetActv.dW12;
end

%% Gradient of biases in both layers
dBias1 = mean(NetActv.r1Pos, 2) - mean(NetActv.r1Neg, 2);
NetActv.dBias1 = learnRate * dBias1 + lambdaMomentum * NetActv.dBias1;

dBias2 = mean(NetActv.r2Pos, 2) - mean(NetActv.r2Neg, 2);
NetActv.dBias2 = learnRate * dBias2 + lambdaMomentum * NetActv.dBias2;

%% Updating BM parameters

BMStruct.W22 = BMStruct.W22 + NetActv.dW22;

% if BMStruct.bLearnW11
%     BMStruct.W11 = BMStruct.W11 + NetActv.dW11;
% end
% 
if BMStruct.bLearnW12
    BMStruct.W12 = BMStruct.W12 + NetActv.dW12;
end

BMStruct.Bias1 = BMStruct.Bias1 + NetActv.dBias1;
BMStruct.Bias2 = BMStruct.Bias2 + NetActv.dBias2;

end