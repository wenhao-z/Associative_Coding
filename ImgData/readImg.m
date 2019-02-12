function buf = readImg(fileName)
% Read images from Van Heteran's natural image database
% More information about this database can be found as 
% http://www.kyb.tuebingen.mpg.de/?id=227s

% Wen-Hao Zhang, Sep-5, 2016

% fileName = fileList(iter).name;
f1 = fopen(fileName,'rb','ieee-be');
w=1536;
h=1024;
buf = fread(f1,[w,h],'uint16');
buf = buf';

fclose(f1);