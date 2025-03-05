import numpy as np
import matplotlib.pyplot as plt
from vmdpy import VMD
import pandas as pd
# 读取Excel文件中的数据
excel_file = 'DATA.xlsx'
df = pd.read_excel(excel_file)

# 提取第一列数据
f = df.iloc[:, 0]
T = len(f)
t = np.arange(1,T+1)
#f = v_1 + v_2 + v_3 + 0.1*np.random.randn(v_1.size)
f_hat = np.fft.fftshift((np.fft.fft(f)))

# some sample parameters for VMD
alpha = 2000       # moderate bandwidth constraint
tau = 0.            # noise-tolerance (no strict fidelity enforcement)
K = 9              # 3 modes
DC = 0             # no DC part imposed
init = 1           # initialize omegas uniformly
tol = 1e-7

# Run actual VMD code
u, u_hat, omega = VMD(f, alpha, tau, K, DC, init, tol)

#%%
# Simple Visualization of decomposed modes
plt.figure(figsize=(10, 8))
plt.plot(u.T)
plt.title('Decomposed modes')

# For convenience here: Order omegas increasingly and reindex u/u_hat
sortIndex = np.argsort(omega[-1,:])
omega = omega[:,sortIndex]
u_hat = u_hat[:,sortIndex]
u = u[sortIndex,:]
#linestyles = ['k', 'k', 'k', 'k', 'k', 'k', 'k','g ']
print(len(u[0,:]))

for k in range(K+1, 0, -1):
    plt.subplot(K + 1, 1, K - k + 2)  # 创建一个子图，行数为K+2，当前子图索引为K-k+2
    if k == K+1:
        plt.plot(t, f, color='#BC4018', linewidth=0.8)
        plt.ylabel('Original',va='bottom',fontsize=13)
        continue
    if K-k+1 == 4:
        plt.plot(t, u[k - 1, :], 'k', linewidth=0.8)
        plt.ylabel('IMF%d' % (K - k + 1), va='center', fontsize=13)  # 修改标签顺序
        continue
    if K-k+1 == 8:
        plt.plot(t, u[k - 1, :], 'k', linewidth=0.8)
        plt.ylabel('IMF%d' % (K - k + 1), va='center', fontsize=13)  # 修改标签顺序
        continue
    plt.plot(t, u[k-1, :], 'k', linewidth=0.8)
    plt.ylabel('IMF%d' % (K - k + 1),va='bottom',fontsize=13)  # 修改标签顺序
plt.savefig("EUA分解.png")
plt.show()
# 将数据转换为DataFrame对象
u=u.T
df = pd.DataFrame(u)

#定义要导出的Excel文件名
excel_file = 'VMD分解结果.xlsx'

#将DataFrame对象写入Excel文件
df.to_excel(excel_file, index=False, header=False)
print(f"数据已成功导出到 {excel_file}")


