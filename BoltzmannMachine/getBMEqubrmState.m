function Xss = getBMEqubrmState(X0, W, Iext, MdlPars)
% Get the equilibrium state of mean field Boltzmann machine
% This fuctions solves the equation
%       x = sigm(Wx + b)
% by iterating following differential equation to its stable state
%      dx/dt = -x + sigm(Wx + b);
% where sigm is the sigmoid function sigm(x) = 1/ (1 + exp(-x));

% Inputs
% X0: initial state of the activities of neurons. [n, 1] column vector.
% W: connection matrix of a Boltzmann machine.
%    W is a symmetric [n, n] matrix
% Iext: external inputs (either feedforward or feedback). [n, 1] column vector;
% MdlPars: a struct storing model parameters.

% Outputs
% Xss: steady state of neurons' activations. [n, 1] column vector.

% Details refer to
% Welling et al., A New Learning Algorithm for Mean Field Boltzmann Machines

% Wen-Hao Zhang, Sep-9, 2016
% @Carnegie Mellon University

dX = 1e2 * MdlPars.tolDX;
X = X0;
nIter = 1;

while (mean(abs(dX(:))) > MdlPars.tolDX) && (nIter < MdlPars.maxIter)
    dX = -X + 1./(1+exp(-(W*X + Iext)) );
    X = X + MdlPars.coefIter * dX;
    nIter = nIter + 1;
end

Xss = X;

end