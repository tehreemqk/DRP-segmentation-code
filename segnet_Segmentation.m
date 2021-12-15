
clear all

close all
clc
sourceDataLoc = "Unet_1";
net1 = vgg16();

%trainedUnet_url = 'https://www.mathworks.com/supportfiles/vision/data/multispectralUnet.mat';

num_images = 1014;
trdir = fullfile(sourceDataLoc,'Tr_data\orginal_image');
trds  = imageDatastore(trdir,'FileExtensions','.tif')
gtdir = fullfile(sourceDataLoc,'Tr_data\gtruth_image')
classNames   = ["pores"                              
                "quarts"
                          ];
pixelLabelID = [0 
                  1];
pxds         = pixelLabelDatastore(gtdir,classNames,pixelLabelID)

I = readimage(trds,num_images);
C = readimage(pxds, num_images);
cmap = bloodSmearColorMap;
B = labeloverlay(I,C,'ColorMap',cmap);

figure(1),title('percentage values shows')
imshow(B)
pixelLabelColorbar(cmap,classNames);
tbl = countEachLabel(pxds)
frequency = tbl.PixelCount/sum(tbl.PixelCount);

figure(2)
bar(1:numel(classNames),frequency)
xticks(1:numel(classNames))
xticklabels(tbl.Name)
xtickangle(45)
ylabel('Frequency')
imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq
pxLayer = pixelClassificationLayer('Name','labels','ClassNames', tbl.Name, 'ClassWeights', classWeights)
imageSize = [256 256 3];
numClasses = numel(classNames );
lgraph = segnetLayers(imageSize,numClasses,'vgg16');
lgraph = removeLayers(lgraph, 'pixelLabels');
lgraph = addLayers(lgraph, pxLayer);
lgraph = connectLayers(lgraph, 'softmax' ,'labels');
%????SegNet??????
figure(3)
plot(lgraph)
options = trainingOptions('sgdm', ...
    'Momentum', 0.9, ...
    'InitialLearnRate', 1e-3, ...
    'L2Regularization', 0.0005, ...
    'MaxEpochs', 16, ...
    'MiniBatchSize', 1, ...
    'Shuffle', 'every-epoch', ...
    'Plots','training-progress', ...
    'VerboseFrequency', 1000);

%augmenter = imageDataAugmenter('RandXReflection',true,'RandYReflection',true,...
 %   'RandXTranslation', [-10 10], 'RandYTranslation', [-10 10], 'RandRotation', [-180 180]);
%datasource = pixelLabelImageSource(trds,pxds,...
  %'DataAugmentation',augmenter)

datasource = pixelLabelImageSource(trds,pxds);

[net, info] = trainNetwork(datasource,lgraph,options)

testdir = fullfile(sourceDataLoc,'Test_data\orginal_image');
testds  = imageDatastore(testdir,'FileExtensions','.tif')
idx = 1014;
sum_quarts = 0;
%I = readimage(testds,i)
for i= 1:idx
    I = readimage(testds,i);
    C = semanticseg(I, net);
    B = labeloverlay(I, C, 'Colormap', cmap, 'Transparency',0.4);
   imshowpair(I, B, 'montage')
  pixelLabelColorbar(cmap, classNames);
  testdir = fullfile(sourceDataLoc,'Test_data\gtruth_image')
 classNames   = ["pores"                              
                "quarts"
                          ];
   pixelLabelID = [0
                1];
   pxdstest        = pixelLabelDatastore(testdir,classNames,pixelLabelID)
  expectedResult = readimage(pxdstest,idx);
  actual = uint8(C);
  expected = uint8(expectedResult);
  figure(idx)
  imshowpair(actual, expected,'montage')
  iou = jaccard(C, expectedResult);
%classNames = 'pore' ./ 'nonpore'
  table(classNames,iou)
  pxdsResults = semanticseg(testds,net,'WriteLocation',tempdir,'Verbose',false);
  metrics = evaluateSemanticSegmentation(pxdsResults,pxdstest,'Verbose',false);
  metrics.DataSetMetrics
  metrics.ClassMetrics
  total=numel(actual)
  pores=length(actual(actual==1))
  quarts=length(actual(actual==2))
   quarts_per=quarts /total
  sum_quarts=sum_quarts+quarts_per
end
total_per = sum_quarts/1014


