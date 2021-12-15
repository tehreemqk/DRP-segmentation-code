clear all
close all
imfile='tif ';
ist=1;
incr=1;
iend=0;
sum_porosity=0;
images=dir([imfile '\*.tif']);
N=natsortfiles({images.name});
%S = dir(fullfile(D,'Carb\*.tif')); % pattern to match filenames.
for k = ist:incr:iend
    k
    

    F = [imfile '\' N{k}]
    I = imread(F);
    %imshow(I)
   % I = imread('coins.png');
%Calculate a threshold using graythresh. The threshold is normalized to the range [0, 1].

level = graythresh(I);
%level = 0.4941
%Convert the image into a binary image using the threshold.
B2 = medfilt2(b,[5 5]);
b = imbinarize(I,0.81);  % Insert threshold for a particular mineral
%B2 = medfilt2(b,[5 5]); % median filter to decrease noise
%B2 = bwareaopen(B2,10);
%imshow(B2,[]);
pixels=numel(B2)
zero=length(B2(B2==0));
nonzero=length(B2(B2~=0))
porosity=zero/pixels
sum_porosity=porosity+sum_porosity
end
num=iend+ist-1
avg_porosity=sum_porosity/num