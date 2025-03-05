import numpy as np
import math
import pandas as pd
from lssvr import LSSVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

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

# 读取数据并进行归一化
xlsx_file = 'EUA.xlsx'  # Excel文件名
sheet_name = 'Sheet1'    # 工作表名称
milk = pd.read_excel(xlsx_file, sheet_name=sheet_name, usecols=[12]) #想预测哪一列就调整这个数字

# 假设time_series_data是一个包含时间序列数据的numpy数组
data_length = len(milk)
input_length = 3  # 输入的历史值数量
output_length = 1  # 输出的未来值数量

# 创建输入和输出序列
X, y = [], []
for i in range(data_length - input_length - output_length + 1):
    X.append(milk[i:i+input_length])
    y.append(milk[i+input_length:i+input_length+output_length])

# 将列表转换为numpy数组
X = np.array(X)
y = np.array(y)

# 假设你希望将数据划分为90％的训练集和10％的测试集
train_size = int(0.9 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
X_train = np.reshape(X_train, (X_train.shape[0], 3*1))
X_test = np.reshape(X_test, (X_test.shape[0], 3*1))
y_train = np.reshape(y_train, (y_train.shape[0], 1))
y_test = np.reshape(y_test, (y_test.shape[0], 1))
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

model = LSSVR(C=60,kernel='rbf',gamma=0.0001)
model.fit(X_train, y_train)
y_hat = model.predict(X_test)
trainresult=model.predict(X_train)

# 计算预测结果的评估指标
plot_actual_vs_predicted(y_train, trainresult, 'Training Set: Actual vs. Predicted')
plot_actual_vs_predicted(y_test, y_hat, 'Test Set: Actual vs. Predicted')

# 计算预测结果的评估指标
rmse = format(RMSE(y_test, y_hat), '.4f')
mape = format(MAPE(y_test, y_hat), '.4f')
r2 = format(r2_score(y_test, y_hat), '.4f')
mae = format(mean_absolute_error(y_test, y_hat), '.4f')
print('RMSE:' + str(rmse) + '\n' + 'MAE:' + str(mae) + '\n' + 'MAPE:' + str(mape) + '\n' + 'R2:' + str(r2))

# 定义要导出的Excel文件名
dftrainy = pd.DataFrame(trainresult)
dftesty = pd.DataFrame(y_hat)
excel_file1 = 'EUA_LSSVM训练集y.xlsx'
excel_file2 = 'EUA_LSSVM预测值.xlsx'
# 将DataFrame对象写入Excel文件
dftrainy.to_excel(excel_file1, index=False, header=False)
print(f"数据已成功导出到 {excel_file1}")
dftesty.to_excel(excel_file2, index=False, header=False)
print(f"数据已成功导出到 {excel_file2}")