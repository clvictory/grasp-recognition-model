% A simple demo  for constructing, training and testing the grasp
% recognition model

% Author: Lu Chen

digitDatasetPath = 'trainDataRGB';
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

% divide the data into training, validation and testing set
[imdsTrain,imdsValidation,imdsTest] = splitEachLabel(imds,0.6,0.1,0.3,'randomize');

% image augmentation
imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-20,20], ...
    'RandXTranslation',[-3 3], ...
    'RandYTranslation',[-3 3]);

imageSize = [24 24 3];
augimds = augmentedImageDatastore(imageSize,imds,'DataAugmentation',imageAugmenter);

% model architecture
layers = [
    imageInputLayer([24 24 3])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
       
    fullyConnectedLayer(100)
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

% training options
options = trainingOptions('sgdm', ...
    'MaxEpochs',10, ...
    'ValidationData',imdsValidation, ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.2,...
    'LearnRateDropPeriod',3,...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

% calculating accuracy on testing set
[graspNet,trainInfo] = trainNetwork(augimds,layers,options);
YPred = classify(graspNet,imdsTest);
YTest = imdsTest.Labels;
accuracy = sum(YPred == YTest)/numel(YTest);
fprintf('Accuracy on testing set is: %.1f%%\n',accuracy*100);

% confusion matrix
% plotconfusion(YTest,YPred);
