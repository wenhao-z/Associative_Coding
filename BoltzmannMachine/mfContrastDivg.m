function NetActv = mfContrastDivg(BMStruct, NetActv)
% Calculate the mean field contrastive divergence to update the state of
% Boltzmann machine

% Wen-Hao Zhang, Sep-22, 2016
% @Carnegie Mellon University

% Wab: connection weight from b -> a.

% Note that W12 may be a row vector if 
% 1) the number of hidden and visiable units are the same;
% 2) every feedforward connections have the same weight


%% Positive phase (clamped with data)
% Solve the mean activation of HIDDEN units given visible inputs (r1Pos)
% NOTE: W12 is a row vector because it is a diagonal matrix.
Iff = bsxfun(@times, BMStruct.W12', NetActv.r1Pos);
Iff = bsxfun(@plus, Iff, BMStruct.Bias2);
r2Pos = 1./(1 + exp(-Iff)); % use the activation when only receiving feedforward inputs as initialization
r2Pos = getBMEqubrmState(r2Pos, BMStruct.W22, Iff, BMStruct);

%% Negative phase (evolving by using its own dynamics)
r1Neg = NetActv.r1Pos;
r2Neg = r2Pos;
for iterCD = 1: BMStruct.maxNumCD
    % Feedback inputs from layer 2 (hidden layer) to layer 1 (visible layer)
    Ifb = bsxfun(@times, BMStruct.W12', r2Neg);
    Ifb = bsxfun(@plus, Ifb, BMStruct.Bias1);
    r1Neg = getBMEqubrmState(r1Neg, BMStruct.W11, Ifb, BMStruct);
    
    % Feedforward inputs from layer 1 (visible layer) to layer 2 (hidden layer)
    Iff = bsxfun(@times, BMStruct.W12', r1Neg);
    Iff = bsxfun(@plus, Iff, BMStruct.Bias2);
    r2Neg = getBMEqubrmState(r2Neg, BMStruct.W22, Iff, BMStruct);
end


%% Folding parameters
NetActv.r2Pos = r2Pos;
NetActv.r1Neg = r1Neg;
NetActv.r2Neg = r2Neg;

end