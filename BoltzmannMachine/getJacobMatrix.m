function JacobMat = getJacobMatrix(BMStruct, rFP)
% Calculate the Jacobian matrix around a fixed point of a mean field
% Boltzmann machine

% INPUT:
% BMStruct: a struct storing the parameters of a Boltzmann machine
% rFP: the mean stationary firing rate of HIDDEN neurons, [BMStruct.numHiddenNeuron, 1]

% OUTPUT:
% JacobMat: [BMStruct.numHiddenNeuron, BMStruct.numHiddenNeuron]

% Wen-Hao Zhang, Oct-11, 2016
% wenhaoz1@andrew.cmu.edu
% @Carnegie Mellon University

dIdY = exp(-rFP) ./ (1 + exp(-rFP)).^2;
JacobMat = bsxfun(@times, dIdY, BMStruct.W22');

JacobMat = -( eye(BMStruct.numHiddenNeuron) - JacobMat );


