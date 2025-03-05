import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from tcn import TCN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 计算均方根误差（Root Mean Square Error，RMSE）
def RMSE(test, predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    return rmse

# 计算平均绝对百分比误差（Mean Absolute Percentage Error，MAPE）
def MAPE(Y_true, Y_pred):
    return np.mean(np.abs((Y_true - Y_pred) / Y_true)) * 100

# 创建滑动窗口的时间序列数据集，用于监督学习问题
def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)

# 绘制实际值和预测值的图表
def plot_actual_vs_predicted(actual, predicted, title):
    plt.figure(figsize=(10, 6))
    plt.plot(actual, label='Actual', color='blue')
    plt.plot(predicted, label='Predicted', color='red')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# 读取数据并进行归一化
xlsx_file = 'IV-IC-IC.xlsx'  # Excel文件名
sheet_name = 'Sheet1'    # 工作表名称
milk = pd.read_excel(xlsx_file, sheet_name=sheet_name, usecols=[15]) #想预测哪一列就调整这个数字
dataset = milk.values
dataset = dataset.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
lookback_window = 3  # 滑动窗口大小，代表历史时间序列的长度

# 划分训练集和测试集
train_size = int(len(dataset) * 0.9)
train, test = dataset[0:train_size], dataset[train_size:]
trainX, trainY = create_dataset(train, lookback_window)
testX, testY = create_dataset(test, lookback_window)
trainX = np.reshape(trainX, (trainX.shape[0], lookback_window, 1))
testX = np.reshape(testX, (testX.shape[0], lookback_window, 1))
print(trainX.shape, trainY.shape, testX.shape, testY.shape)
i = Input(shape=(lookback_window, 1))
m = TCN()(i)
m = Dense(1, activation='linear')(m)  # 输出维度改为1，因为预测的是一个值而不是12个值
model = Model(inputs=[i], outputs=[m])
model.summary()
model.compile('adam', 'mae')
print('Train...')
model.fit(trainX, trainY, epochs=200, verbose=2, batch_size=64)
result = model.predict(testX)
resultx = model.predict(trainX)
result = scaler.inverse_transform(result)
testY = scaler.inverse_transform(testY)
resultx = scaler.inverse_transform(resultx)
trainY = scaler.inverse_transform(trainY)

# 计算预测结果的评估指标
rmse = format(RMSE(testY, result), '.4f')
mape = format(MAPE(testY, result), '.4f')
r2 = format(r2_score(testY, result), '.4f')
mae = format(mean_absolute_error(testY, result), '.4f')
print('RMSE:' + str(rmse) + '\n' + 'MAE:' + str(mae) + '\n' + 'MAPE:' + str(mape) + '\n' + 'R2:' + str(r2))

# 定义要导出的Excel文件名
dftrainy = pd.DataFrame(resultx)
dftesty = pd.DataFrame(result)
excel_file1 = 'EUA_TCN训练集y.xlsx'
excel_file2 = 'EUA_TCN预测值.xlsx'

# 将DataFrame对象写入Excel文件
dftrainy.to_excel(excel_file1, index=False, header=False)
print(f"数据已成功导出到 {excel_file1}")
dftesty.to_excel(excel_file2, index=False, header=False)
print(f"数据已成功导出到 {excel_file2}")

# 绘制训练集和测试集的实际值和预测值图表
plot_actual_vs_predicted(trainY, resultx, 'Training Set: Actual vs. Predicted')
plot_actual_vs_predicted(testY, result, 'Test Set: Actual vs. Predicted')