#目的:实施Diebold-Mariano检验(DM检验)进行比较预测精度
#1) actual_lst:实际值的列表
# 2) pred1_lst:预测值的第一个列表
# 3) pred2_lst:第二个预测值列表
# 4) :param h: 预测模型是几步预测，h就是几
# 5) crit:一个字符串指定标准
# i) MSE:均方误差
# ii) MAD:平均绝对偏差
# iii) MAPE:平均绝对百分比误差
# iv)多:用幂函数来衡量误差
#poly:用于暴击的能量
#(只有当暴击为“poly”时才有意义)
#条件:1)actual_lst, pred1_lst和pred2_lst的长度相等
# 2) h必须是一个整数,它必须大于0小于actual_lst的长度
# 3) critical必须接受Input中指定的4个值
# 4) actual_lst, pred1_lst和pred2_lst的每个值必须
#为数值。缺少的值将不被接受。
# 5)权力必须是一个数值。
#返回:2个元素的命名元组
#p_value: DM测试的p值
# 2) DM: DM测试的测试统计数据
##########################################################
#引用:
# Harvey, D., Leybourne, S., & Newbold, P. (1997).  Testing the equality of
#   prediction mean squared errors.  International Journal of forecasting,
#   13(2), 281-291.
#
# Diebold, F. X. and Mariano, R. S. (1995), Comparing predictive accuracy,
#   Journal of business & economic statistics 13(3), 253-264.
#
def dm_test(actual_lst, pred1_lst, pred2_lst, h = 1, crit="MSE", power = 2):
    # Routine for checking errors
    def error_check():
        rt = 0
        msg = ""
        # Check if h is an integer
        if (not isinstance(h, int)):
            rt = -1
            msg = "The type of the number of steps ahead (h) is not an integer."
            return (rt,msg)
        # Check the range of h
        if (h < 1):
            rt = -1
            msg = "The number of steps ahead (h) is not large enough."
            return (rt,msg)
        len_act = len(actual_lst)
        len_p1  = len(pred1_lst)
        len_p2  = len(pred2_lst)
        # Check if lengths of actual values and predicted values are equal
        if (len_act != len_p1 or len_p1 != len_p2 or len_act != len_p2):
            rt = -1
            msg = "Lengths of actual_lst, pred1_lst and pred2_lst do not match."
            return (rt,msg)
        # Check range of h
        if (h >= len_act):
            rt = -1
            msg = "The number of steps ahead is too large."
            return (rt,msg)
        # Check if criterion supported
        if (crit != "MSE" and crit != "MAPE" and crit != "MAD" and crit != "poly"):
            rt = -1
            msg = "The criterion is not supported."
            return (rt,msg)  
        # Check if every value of the input lists are numerical values
        from re import compile as re_compile
        comp = re_compile("^\d+?\.\d+?$")  
        def compiled_regex(s):
            """ Returns True is string is a number. """
            if comp.match(s) is None:
                return s.isdigit()
            return True
        for actual, pred1, pred2 in zip(actual_lst, pred1_lst, pred2_lst):
            is_actual_ok = compiled_regex(str(abs(actual)))
            is_pred1_ok = compiled_regex(str(abs(pred1)))
            is_pred2_ok = compiled_regex(str(abs(pred2)))
            if (not (is_actual_ok and is_pred1_ok and is_pred2_ok)):  
                msg = "An element in the actual_lst, pred1_lst or pred2_lst is not numeric."
                rt = -1
                return (rt,msg)
        return (rt,msg)
    
    # Error check
    error_code = error_check()
    # Raise error if cannot pass error check
    if (error_code[0] == -1):
        raise SyntaxError(error_code[1])
        return
    # Import libraries
    from scipy.stats import t
    import collections
    import pandas as pd
    import numpy as np
    
    # Initialise lists
    e1_lst = []
    e2_lst = []
    d_lst  = []
    
    # convert every value of the lists into real values
    actual_lst = pd.Series(actual_lst).apply(lambda x: float(x)).tolist()
    pred1_lst = pd.Series(pred1_lst).apply(lambda x: float(x)).tolist()
    pred2_lst = pd.Series(pred2_lst).apply(lambda x: float(x)).tolist()
    
    # Length of lists (as real numbers)
    T = float(len(actual_lst))
    
    # construct d according to crit
    if (crit == "MSE"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append((actual - p1)**2)
            e2_lst.append((actual - p2)**2)
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "MAD"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append(abs(actual - p1))
            e2_lst.append(abs(actual - p2))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "MAPE"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append(abs((actual - p1)/actual))
            e2_lst.append(abs((actual - p2)/actual))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "poly"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append(((actual - p1))**(power))
            e2_lst.append(((actual - p2))**(power))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)    
    
    # Mean of d        
    mean_d = pd.Series(d_lst).mean()
    
    # Find autocovariance and construct DM test statistics
    def autocovariance(Xi, N, k, Xs):
        autoCov = 0
        T = float(N)
        for i in np.arange(0, N-k):
              autoCov += ((Xi[i+k])-Xs)*(Xi[i]-Xs)
        return (1/(T))*autoCov
    gamma = []
    for lag in range(0,h):
        gamma.append(autocovariance(d_lst,len(d_lst),lag,mean_d)) # 0, 1, 2
    V_d = (gamma[0] + 2*sum(gamma[1:]))/T
    DM_stat=V_d**(-0.5)*mean_d
    harvey_adj=((T+1-2*h+h*(h-1)/T)/T)**(0.5)
    DM_stat = harvey_adj*DM_stat
    # Find p-value
    p_value = 2*t.cdf(-abs(DM_stat), df = T - 1)
    print("%.4f" %DM_stat)
    # Construct named tuple for return
    dm_return = collections.namedtuple('dm_return', 'DM p_value')
    rt = dm_return(DM = DM_stat, p_value = p_value)
    return rt

from dm_test import dm_test
import random
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import datetime
import xlrd
file_location = "C:\\Users\\DELL\\Desktop\\CN价格.xls"
data = xlrd.open_workbook(file_location)
sheet = data.sheet_by_index(0)
actual_lst = [sheet.cell_value(r,0) for r in range(1,sheet.nrows)]  # x 0代表第一列，1代表第二列
pred1_lst = [sheet.cell_value(r,20) for r in range(1,sheet.nrows)]  # y
pred2_lst = [sheet.cell_value(r,19) for r in range(1,sheet.nrows)]
print(actual_lst)
print(pred1_lst)
print(pred2_lst)

print(len(pred1_lst))
pmse = dm_test(actual_lst,pred1_lst,pred2_lst,h =1, crit="MSE")
pmae = dm_test(actual_lst,pred1_lst,pred2_lst,h =1, crit="MAD")
pmape=dm_test(actual_lst,pred1_lst,pred2_lst,h =1, crit="MAPE")
print(pmse,pmae,pmape)