% Set the working path of associative coding model

% Wen-Hao Zhang, Sep-7, 2016
% @Carnegie Mellon University

% The root in my pc at CNBC-CMU
Path_RootDir = fileparts(mfilename('fullpath'));

Idx = strfind(Path_RootDir, '/');
Path_DataBase = fullfile(Path_RootDir(1: Idx(end-1)), 'Database', 'vanHateren');
clear Idx

addpath(Path_RootDir);
addpath(fullfile(Path_RootDir, 'ModelParams'));
addpath(fullfile(Path_RootDir, 'BoltzmannMachine'));
addpath(fullfile(Path_RootDir, 'GaborWavelets'));
addpath(fullfile(Path_RootDir, 'plotCodes'));
addpath(fullfile(Path_RootDir, 'ImgData'));
addpath(Path_DataBase);
cd(Path_RootDir)
