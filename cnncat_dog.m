
digitDatasetPath = fullfile('C:\','Users','Shreya Kohli','Desktop','project','train');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
inputSize = [300 300 3];
% imds.ReadFcn = @(loc)imresize(rgb2gray(loc));
dsTrain = transform(imds,@commonPreprocessing);
imds.ReadFcn = @(loc)imresize(imread(loc),[300 300 ]);
numTrainFiles = 1000;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');
%dsTrain = transform(imdsTrain,@commonPreprocessing);
%dsValidation = transform(imdsValidation,@commonPreprocessing);
%inputSize = [100 100 3];
numClasses = 2;

lgraph = layerGraph;
% layers = [
%     imageInputLayer([200 200 3],'Name','input')
%     convolution2dLayer(3,32,'Name','conv_1','Padding','same')
%     batchNormalizationLayer('Name','bat_norm_1') 
%     reluLayer('Name','relu_1')
%    
%     maxPooling2dLayer(2,'Stride',2)
%     
%     convolution2dLayer(3,64,'Name','conv_1','Padding','same')
%     batchNormalizationLayer('Name','bat_norm_1') 
%     reluLayer('Name','relu_2')
%  
%     maxPooling2dLayer(2,'Stride',2)
%       convolution2dLayer(3,128,'Name','conv_1','Padding','same')
%        batchNormalizationLayer('Name','bat_norm_1') 
%        reluLayer('Name','relu_2')
%      
%    % 
% %     batchNormalizationLayer('Name','bat_norm_2')
% %     reluLayer('Name','relu_2')
%  
%     fullyConnectedLayer(numClasses,'Name','fullcl_1')
%    softmaxLayer('Name','soft_1')
%     classificationLayer('Name','final')];
layers = [
    imageInputLayer([300 300 3])

    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    averagePooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    averagePooling2dLayer(2,'Stride',2)
  
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    dropoutLayer(0.2)
    fullyConnectedLayer(2)
       softmaxLayer('Name','soft_1')
    classificationLayer('Name','final')];
% % options = trainingOptions('sgdm', ...
% %     'MaxEpochs',4, ...
% %     'ValidationData',imdsValidation, ...
% %     'ValidationFrequency',30, ...
% %     'Verbose',false, ...
% %     'Plots','training-progress');
miniBatchSize  = 128;
% validationFrequency = floor(numel(YTrain)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',30, ...
    'InitialLearnRate',1e-3, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',20, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Plots','training-progress', ...
    'Verbose',false);

net = trainNetwork(imdsTrain,layers,options);
%%
aa=imread('104.jpg');
img = imresize(aa,[100 100]);
[classfn,score] = classify(net,img);
imshow(img);
title(sprintf("%s (%.2f)", classfn, score(classfn)));
%%YPred = classify(net,imdsValidation);
%%YValidation = imdsValidation.Labels;
%%accuracy = mean(YPred == YValidation)
%%YPred = classify(net,aa);
%%
lgraph = layerGraph;
lgraph = addLayers(lgraph,layers);
layers(1).Mean=4;
figure
plot(lgraph)
lgraph = removeLayers(lgraph, lgraph.Layers(end).Name);
dlnet = dlnetwork(lgraph);
softmaxName = 'soft_1';
convLayerName = 'conv_2';
dlImg = dlarray(single(img),'SSC');
[convMap, dScoresdMap] = dlfeval(@gradcam, dlnet, dlImg, softmaxName, convLayerName, classfn);
gradcamMap = sum(convMap .* sum(dScoresdMap, [1 2]), 3);
gradcamMap = extractdata(gradcamMap);
gradcamMap = rescale(gradcamMap);
gradcamMap = imresize(gradcamMap, [100 100], 'Method', 'bicubic');

imshow(img);
hold on;
imagesc(gradcamMap,'AlphaData',0.5);
colormap jet
hold off;
title("Grad-CAM");