%clear;clc;
%load('gzt.mat')
%% 构造样本集
% 数据个数
gzt=gzt';
n=length(gzt);

% 确保Z为列向量gzt=gzt(:);


% Z(n) 由Z(n-1),Z(n-2),...,Z(n-L)共L个数预测得到.
L = 3;

% Z_n：每列为一个构造完毕的样本，共n-L个样本
Z_n = zeros(L+1, n-L);
for i=1:n-L
    Z_n(:,i) = gzt(i:i+L);
end
numdata=floor(0.8*n);
%% 划分训练集测试集
trainx = Z_n(1:3, 1:numdata-3);
trainy = Z_n(4, 1:numdata-3); %多步预测或者是单步预测

testx = Z_n(1:3, numdata-2:end);
testy = Z_n(4, numdata-2:end);
%% 数据预处理
[trainx1, st1] = mapminmax(trainx);
[trainy1, st2] = mapminmax(trainy);  

% 测试数据做与训练数据相同的归一化操作
testx1 = mapminmax('apply',testx,st1);
testy1 = mapminmax('apply',testy,st2);
%% 一维特征lstm网络训练
numFeatures = 3;   %特征维
numResponses = 1;  %输出维
numHiddenUnits = 100;   %创建LSTM回归网络，指定LSTM层的隐含单元个数。可调
 
layers = [ ...
    sequenceInputLayer(numFeatures)    %输入层
    gruLayer(numHiddenUnits)  % lstm层，如果是构建多层的LSTM模型，可以修改。
    fullyConnectedLayer(numResponses)    %为全连接层,是输出的维数。
    regressionLayer];      %其计算回归问题的半均方误差模块 。即说明这不是在进行分类问题。
 
%指定训练选项，求解器设置为adam， 1000轮训练。
%梯度阈值设置为 1。指定初始学习率 0.01，在 125 轮训练后通过乘以因子 0.2 来降低学习率。
options = trainingOptions('adam', ...
    'MaxEpochs',200, ...
    'GradientThreshold',1, ...
    'MiniBatchSize',128, ...
    'InitialLearnRate',0.005, ...      
    'LearnRateSchedule','piecewise', ...%每当经过一定数量的时期时，学习率就会乘以一个系数。
    'LearnRateDropPeriod',400, ...      %乘法之间的纪元数由“ LearnRateDropPeriod”控制。可调
    'LearnRateDropFactor',0.15, ...      %乘法因子由参“ LearnRateDropFactor”控制，可调
    'Verbose',0,  ...  %如果将其设置为true，则有关训练进度的信息将被打印到命令窗口中。默认值为true。
    'Plots','none');    %构建曲线图 将'training-progress'替换为none
net = trainNetwork(trainx1,trainy1,layers,options); 
%% 海鸥优化
% 海鸥优化算法对全连接层权重进行优化
numSeagulls = 5; % 海鸥个数
maxIterations = 10; % 最大迭代次数

% 获取全连接层的权重向量
fcLayerIndex = 3;
fcLayerWeights = net.Layers(fcLayerIndex).Weights(:);

% 初始化海鸥位置
seagulls = (0.5 - rand(numSeagulls, numel(fcLayerWeights))) * 2; % 随机初始化权重在[-1, 1]范围内

% 进行海鸥优化算法优化
for iter = 1:maxIterations
    % 计算当前位置的损失值
    values = zeros(numSeagulls, 1);
    for i = 1:numSeagulls
        weights = fcLayerWeights + seagulls(i, :)'; % 将海鸥位置加到当前权重上
        
        % 创建一个新的全连接层，并使用当前权重
        newFcLayer = fullyConnectedLayer(numResponses, 'Weights', reshape(weights, size(net.Layers(fcLayerIndex).Weights)));
        tempLayers = layers;
        tempLayers(fcLayerIndex) = newFcLayer;
        
        % 使用当前权重进行训练并预测
        tempNet = trainNetwork(trainx1, trainy1, tempLayers, options);
        tempNet = predictAndUpdateState(tempNet, trainx1);
        tempTrainTy1 = zeros(size(trainy1));
        numTrain = numel(trainy1);
        for j = 1:numTrain
            [tempNet, tempTrainTy1(j)] =  predictAndUpdateState(tempNet, trainx1(:,j),'ExecutionEnvironment','cpu');
        end
        tempTrainTy = mapminmax('reverse', tempTrainTy1, st2);
        
        % 计算损失
        values(i) = mse(tempTrainTy - trainy1);
    end

    % 更新最佳位置和最佳值
    [bestValue, bestIdx] = min(values);
    bestFCWeights = fcLayerWeights + seagulls(bestIdx, :)';

    % 使用海鸥优化算法更新海鸥位置
    for i = 1:numSeagulls
        if values(i) < bestValue
            % 随机选择另一个海鸥
            j = randi(numSeagulls);
            while j == i
                j = randi(numSeagulls);
            end

            % 更新当前海鸥位置
            seagulls(i, :) = seagulls(i, :) + rand(1, numel(fcLayerWeights)) .* (seagulls(j, :) - seagulls(i, :));
        end
    end

    % 更新全连接层权重为最佳权重
    newFcLayer = fullyConnectedLayer(numResponses, 'Weights', reshape(bestFCWeights, size(net.Layers(fcLayerIndex).Weights)));
    layers(fcLayerIndex) = newFcLayer;
end

% 使用优化后的全连接层进行预测
tempNet = trainNetwork(trainx1, trainy1, layers, options);

%% 神经网络初始化
tempNet = predictAndUpdateState(tempNet,trainx1);  %将新的trainx数据用在网络上进行初始化网络状态
%[net,test_ty1] = predictAndUpdateState(net,trainy1(end-1:end)); %用训练的最后一步来进行预测第一个预测值，给定一个初始值。这是用预测值更新网络状态特有的。

%% 进行用于验证神经网络的数据预测（用预测值更新网络状态）
numtrain=numel(trainy1)/numResponses;
for j = 1:numtrain
    [tempNet,train_ty1(:,j)] =  predictAndUpdateState(tempNet,trainx1(:,j),'ExecutionEnvironment','cpu');
end
train_ty = mapminmax('reverse', train_ty1, st2);

numtest = numel(testy1)/numResponses;
for i = 1:numtest  
    [tempNet,test_ty1(:,i)] = predictAndUpdateState(tempNet,testx1(:,i),'ExecutionEnvironment','cpu');  %predictAndUpdateState函数是一次预测一个值并更新网络状态
end
test_ty = mapminmax('reverse', test_ty1, st2);     %使用先前计算的参数对预测去标准化。
%% 验证神经网络

figure(1)
x=1:length(train_ty);

% 显示真实值
plot(x,trainy,'b-');
hold on
% 显示神经网络的输出值
plot(x,train_ty,'r--')

legend('碳价真实值','Elman网络输出值')
title('训练数据的测试结果');

% 显示残差
figure(2)
plot(x, train_ty - trainy)
title('训练数据测试结果的残差')

% 显示均方误差
mse1 = mse(train_ty - trainy);
fprintf('    mse_train = \n     %f\n', mse1)

% 显示相对误差
disp('    训练数据相对误差：')
fprintf('%f  ', (train_ty - trainy)./trainy );
fprintf('\n')

figure(6)
x=1:length(testy);

% 显示真实值
plot(x,testy,'b-');
hold on
% 显示神经网络的输出值
plot(x,test_ty,'r--')

legend('碳价真实值','LSTM网络输出值')
title('测试数据的测试结果');

% 显示残差
figure(7)
k = (test_ty - testy);
plot(x, test_ty - testy)
title('测试数据测试结果的残差')

% 显示均方误差

mse1 = mse(test_ty - testy);
fprintf('    mse_test = \n     %f\n', mse1)
% 显示相对误差
disp('    测试数据相对误差：')
fprintf('%f  ', (test_ty - testy)./testy );
fprintf('\n')

% 保存相对误差
% 保存相对误差
test_re = test_ty - testy;
train_re = train_ty - trainy;

test_error = (test_ty - testy)./testy;
train_error = (train_ty - trainy)./trainy;
predict=test_ty;
%%  相关指标计算
M = size(trainx1, 2);
N = size(testx1, 2);
%  R2
R1 = 1 - norm(train_ty - trainy)^2 / norm(trainy - mean(trainy))^2;
R2 = 1 - norm(test_ty - testy)^2 / norm(testy -  mean(testy))^2;

disp(['训练集数据的R2为：', num2str(R1)])
disp(['测试集数据的R2为：', num2str(R2)])

%  MAE
mae1 = sum(abs(train_ty - trainy)) ./ M ;
mae2 = sum(abs(test_ty - testy)) ./ N ;

disp(['训练集数据的MAE为：', num2str(mae1)])
disp(['测试集数据的MAE为：', num2str(mae2)])

mape=sum(abs((test_ty - testy)./testy)) ./ N ;
disp(['测试集数据的MAPE为：', num2str(mape)])

%  MBE
mbe1 = sum(train_ty - trainy) ./ M ;
mbe2 = sum(test_ty - testy) ./ N ;

disp(['训练集数据的MBE为：', num2str(mbe1)])
disp(['测试集数据的MBE为：', num2str(mbe2)])
train_ty=train_ty';
test_ty=test_ty';