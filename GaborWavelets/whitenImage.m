function ImgPatch = whitenImage(ImgPatch)
% Whitening image patches by using the filters which is the same as
% Ref: Olshausen BA, Field DJ 1997, Vision Research, 37: 3311-3325

% ImgPatch: [x, y, Index of Images]

% Author: Wen-Hao Zhang, June-9, 2017
% wenhaoz1@andrew.cmu.edu
% @Carnegie Mellon University

szImgPatch = size(ImgPatch);
[fx, fy] = meshgrid(-(szImgPatch(1)-1)/2: (szImgPatch(1)-1)/2, ...
    -(szImgPatch(2)-1)/2: (szImgPatch(2)-1)/2);
rho = sqrt(fx.^2 +fy.^2);
f_0 = 0.4 * szImgPatch(1); % cutoff frequency of a lowpass filter
%                            that is combined with the whitening filter
filt = rho.*exp(-(rho/f_0).^4);
filt = fftshift(filt);
filt = filt ./ std(filt(:));

% Whitening
ImgPatch = fft2(ImgPatch);
ImgPatch = bsxfun(@times, ImgPatch, filt);
ImgPatch = real(ifft2(ImgPatch));
