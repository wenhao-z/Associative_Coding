function filterImgPatch = simGaborWaveletsNew(ImgPatch, MdlPars)
% Using Gabor wavelets to filter natural images. The resutls are then 
% fed into a Boltzmann machine

% Inputs
% ImgPatch： [Height, Width, numImgPatchPerImg, numImg]
%          The parameters can be found at AssoCode.Layer0;
% MdlPars: a STRUCT indicates the parameter of encoding model.
%          It is AssoCode.Layer1           

% Outputs
% filterImgPatch: 2D array
%        [down sample height X down sample width X size(gaborArray), numImgPatch]


% Wen-Hao Zhang
% @Carnegie Mellon University, Sep-9, 2016

%% Extract varialbes from struct
intNeighbKerl = MdlPars.intNeighbKerl;
gaborArray = MdlPars.gaborArray; % Cell with size as [numSpatFreq, numOrient]

%% Reshape ImgPatch into a 3D array%     IOStruct.Layer0.Y = ImgPatch;
szImgPatch = size(ImgPatch);
ImgPatch = reshape(ImgPatch, szImgPatch(1), szImgPatch(2), []);

%% Filter image patches by using Gabor wavelets
% Flip gaborArray before using convn code in MATLAB

gaborArray = cellfun(@(x) flip(flip(x,1), 2), gaborArray, 'uniformout', 0);

filterImgPatch = cellfun(@(x) convnds(ImgPatch, x, intNeighbKerl), gaborArray, ...
    'uniformout', 0);

% filterImgPatch = cellfun(@(x) convn(ImgPatch, x, 'valid'), gaborArray, ...
%     'uniformout', 0);

% filterImgPatch is a cell with the same size as gaborArray
% Each element of filterImgPatch is [height, width, numImgPatch]

%% Down sample the filtered image patch
filterImgPatch = cellfun(@(x) x(1:intNeighbKerl(1):end, 1:intNeighbKerl(2):end,:), ...
    filterImgPatch, 'uniformout', 0);

%% Reorganize the shape of filterImgPatch
% Right shift 3 dims, because the element in a cell is a 3D array
% A 5D array [height, width, numImgPatch, numSpatFreq, numOrient]
filterImgPatch = cell2mat(shiftdim(filterImgPatch, -3));
filterImgPatch = permute(filterImgPatch, [1,2,4,5,3]);
filterImgPatch = reshape(filterImgPatch, [], size(filterImgPatch, 5));

%%
    function filterImg = convnds(ImgPatch, Kerl, step)
       % downsampling convolution (convolution with inteval larger than 1)
       
       szKerl = size(Kerl);
       % Index of filterImgPatch
       szfilterImgPatch = (szImgPatch(1:2) - size(gaborArray{1}))./step+1;
       [sub1, sub2] = ndgrid(1:szfilterImgPatch(1), 1: szfilterImgPatch(2));
       sub = mat2cell([sub1(:), sub2(:)], ones(1, prod(szfilterImgPatch)), 2);
       
       ImgReshape = cellfun(@(x) ...
           ImgPatch((x(1)-1)*step(1)+(1:szKerl(1)), (x(2)-1)*step(2)+(1:szKerl(2)),:), ...
           sub, 'uniformout', 0);
       
       filterImg = cellfun(@(Img) convn(Img, Kerl, 'valid'), ImgReshape, 'uniformout', 0);
       filterImg = reshape(filterImg, szfilterImgPatch);
       filterImg = cell2mat(filterImg);
    end

end