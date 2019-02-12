function [ImgPatch, seedSampleImg] = sampleImgPatch(Layer0, Path_DataBase)
% Sample image patches from database

% Inputs
% Layer0: a strucct specifies the paramters of raw inputs
% Path_DataBase: indicates the path of database
%
% Outputs
% ImgPatch: 4D tensor
% [rows of image patch, columns of images patch, number of patches from each image, number of images in the database]

% Wen-Hao Zhang, Sep-5, 2016
% @Carnegie Mellon University

fprintf('Sampling image patches from image database (%d images X %d patches/image = %d patches)...\n', ...
    length(Layer0.fileList), Layer0.numImgPatchPerImg, length(Layer0.fileList)*Layer0.numImgPatchPerImg);
% fprintf('Sampling image patches from image database (%d images)...\n', length(Layer0.fileList));
% fprintf('Sample %d image pathces from each image.\n', Layer0.numImgPatchPerImg);
fprintf('Size of an image patch: [%d, %d].\n', Layer0.sizeImagePatch(1), Layer0.sizeImagePatch(2));

% Initialize ImgPatch
ImgPatch = zeros([Layer0.sizeImagePatch, Layer0.numImgPatchPerImg, length(Layer0.fileList)]);

% Set random stream to generate random numbers
if ~isfield(Layer0, 'seedSampleImg')
    seedSampleImg = round(sum(clock)*1e3);
else
    seedSampleImg = Layer0.seedSampleImg;
end
[rowStream, colStream] = RandStream.create('mrg32k3a','NumStreams',2, ...
    'seed', seedSampleImg);

tStart = clock;
% Sample image patches from raw images in database
for iterFile = 1: length(Layer0.fileList)
%     fprintf('Sample image patch from %d-th image\n', iterFile);
    % Read an image from database
    buf = readImg(fullfile(Path_DataBase, Layer0.fileList(iterFile).name));
    
    % Sample the row and column index of image patches INDEPENDENTLY
    % Idx(1,:): row index;  Idx(2,:): colum index
    Idx = rowStream.randperm(Layer0.sizeRawImage(1)-Layer0.sizeImagePatch(1)+1, ...
        Layer0.numImgPatchPerImg);
    Idx(2,:) = colStream.randperm(Layer0.sizeRawImage(2)-Layer0.sizeImagePatch(2)+1, ...
        Layer0.numImgPatchPerImg);
    
    for iter = 1: size(Idx, 2)
        ImgPatch(:,:, iter, iterFile) = ...
            buf(Idx(1, iter): Idx(1, iter)+Layer0.sizeImagePatch(1)-1, ...
            Idx(2, iter): Idx(2, iter)+Layer0.sizeImagePatch(2)-1);
    end
end
tEnd = clock;

ImgPatch = ImgPatch;
seedSampleImg = seedSampleImg;

fprintf('Sampling image patches finished! It takes %d seconds.\n', etime(tEnd, tStart));