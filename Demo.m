% A simple demo  for constructing, training and testing the grasp
% recognition model

% Author: Lu Chen

digitDatasetPath = 'trainDataRGB';
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

% image augmentation
imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-20,20], ...
    'RandXTranslation',[-3 3], ...
    'RandYTranslation',[-3 3]);

imageSize = [24 24 3];

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

loop_num = 5;
accuracy_list = zeros(1,loop_num);
for i = 1:loop_num
    % divide the data into training, validation and testing set
    [imdsTrain,imdsValid,imdsTest] = splitEachLabel(imds,0.6,0.1,0.3,'randomize');
    
    augimds = augmentedImageDatastore(imageSize,imds,'DataAugmentation',imageAugmenter);
    
    % training options
    options = trainingOptions('sgdm', ...
        'MaxEpochs',10, ...
        'ValidationData',imdsValid, ...
        'LearnRateSchedule','piecewise',...
        'LearnRateDropFactor',0.2,...
        'LearnRateDropPeriod',3,...
        'ValidationFrequency',30, ...
        'Verbose',false);
    
    % calculating accuracy on testing set
    [graspNet,trainInfo] = trainNetwork(imdsTrain,layers,options);
    YPred = classify(graspNet,imdsTest);
    YTest = imdsTest.Labels;
    accuracy = sum(YPred == YTest)/numel(YTest);
    accuracy_list(i) = accuracy;
    fprintf('Accuracy on testing set for loop %d is: %.1f%%\n',i,accuracy*100);
end

fprintf('Average accuracy is: %.1f%%\n',mean(accuracy_list)*100);