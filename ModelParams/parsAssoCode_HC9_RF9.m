% The parameters for associative coding model.

% Wen-Hao Zhang, Sep-5, 2016
% @Carnegie Mellon University

AssoCode.Layer1.sizeKerl         = [9, 9]; % size of 2D Gabor wavelets. An ODD number is STRONGLY recommended to avoid numerical error.
% AssoCode.Layer1.SpatFreqArray    = 4*9/17; % Making the spatial frequency the same as the onse defined in HC5_RF17
AssoCode.Layer1.SpatFreqArray    = 3; % 3 pixels/ cycle (This is redefined under new gaborWaveLets2D.m, Dec-20, 2016)
AssoCode.Layer2.numHyperCol      = [9, 9]; % number of hypercolumns

AssoCode.Layer2.J12 = 0.1; % Shared and fixed connection weight for W12
AssoCode.Layer2.J22 = 1e-4; % Initial value for W22
AssoCode.Layer2.Bias1_Init = -1; % Initial value
AssoCode.Layer2.Bias2_Init = -1; % Initial value

AssoCode.Layer2.coefIter = 0.05; % used to updating X by using Eq: X + coefIter * dX;
AssoCode.Layer2.maxIter  = 1e3; % maximal number of iteraction for mean field dynamics

AssoCode.Layer1.flagNonLincFunc = 2;
% 0: rectified linear function; 
% 1: cumulative distribution on absolute value; 
% 2: cumulative distribution.
% 3: cumulative distribution on RECTIFIED value; 