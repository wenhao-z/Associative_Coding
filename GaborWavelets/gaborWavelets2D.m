function gaborArray = gaborWavelets2D(szGabor, spatfreq, orientation, kOctave)
% Generate a set of Gabor wavelets

% Inputs:
% szGabor: a 1x2 array specifies the number of rows and columns of gabor wavelet;
% spatfreq: a row vector specifies spatial frequency. Unit: pixel/cycle.
% orientation:  a row vector. Unit: radian

% Output:
% gaborArray, cell with size as [length(spatial frequency), length(orientation)]
% Each element of the cell is a 2D Gabor wavelet, with size as [length(X), length(Y)]

% Details can be found in Eq.3 of
% Lee, 1996, IEEE PAMI, Image Representation Using 2D Gabor Wavelets

% Wen-Hao Zhang,Sep-6, 2016

fprintf('Generating a set of Gabor wavelets (%d spatial frequencies and %d orientations).\n', ...
    length(spatfreq), length(orientation));

if nargin < 4
    %         kOctave = pi;  % corresponds to a frequency bandwidth of one octave
    kOctave = 2.5; % corresponds to a frequency bandwidth of 1.5 octave
end

% Generate 2d grid of x and y of a Gabor wavelet
xArray = (1:szGabor(1)) - (szGabor(1)+1)/2;
yArray = (1:szGabor(2)) - (szGabor(2)+1)/2;

% Generate a 4D tensor for a family of Gabor wavelets
% Dim information:
% [length(X), length(Y), length(spatial frequency), length(orientation)]
spatfreq = 2*pi/spatfreq;
[xArray, yArray, omegaArray, thetaArray] = ndgrid(xArray, yArray, spatfreq, orientation);

% Generate Gabor wavelets
gaborArray = gaborFunc(xArray, yArray, omegaArray, thetaArray);

% Normalize gaborArray into zero mean and unit variance
% gaborArray = bsxfun(@minus, gaborArray, mean(mean(gaborArray, 1), 2));
% gaborArray = bsxfun(@rdivide, gaborArray, sqrt(sum(sum(gaborArray.^2, 1), 2))); % normalize by L2 norm
% gaborArray(isnan(gaborArray)) = 1./ sqrt(prod(szGabor)); % when spatfreq = 0, numerical error will happen.

% Reorganize 4D tensor into cells.
% Each element of the cell is a 2D gabor wavelet.
gaborArray = mat2cell(gaborArray, szGabor(1), szGabor(2), ...
    ones(1, length(spatfreq)), ones(1, length(orientation)));
gaborArray = shiftdim(gaborArray, 2);

    function z = gaborFunc(x, y, omega, theta)
        % Gaussian envelope of Gabor function
        gaussEnv = 4*(x.*cos(theta) + y.*sin(theta)).^2 + (-x.*sin(theta) + y.*cos(theta)).^2;
        gaussEnv = exp(- omega.^2/8 / kOctave.^2 .* gaussEnv);
        
        % Sinusoidal part of Gabor function
        cplxPart = omega .* ( x.*cos(theta) + y.*sin(theta));
        cplxPart = cos(cplxPart);
        
        z = gaussEnv .* cplxPart;
    end

end