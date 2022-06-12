import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from sklearn.metrics import r2_score

# np.random.seed(0)

from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

"""计算平均绝对百分误差 (MAPE)"""

def get_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))
#选择模型；'lin','rbf','poly'
model='rbf'

test_size = 0.2  # 用作测试集的数据比例


path = 'surface tension+RDF.xlsx'

df = pd.read_excel(path, encoding='utf-8')

cols_list=df.columns[1:-1]
target = df.columns[-1]

import sklearn.preprocessing
min_max_scaler = sklearn.preprocessing.MinMaxScaler()
for i in range(len(cols_list)):
    df[cols_list[i]] = min_max_scaler.fit_transform(df[cols_list[i]].values.reshape(-1, 1))
df[target] = min_max_scaler.fit_transform(df[target].values.reshape(-1, 1))


num_test = int(test_size * len(df))
num_train = len(df) - num_test
# print(num_train)
train = df[:num_train]  # 959 rows x 7 columns
# print(train)
test = df[num_train:]  # 239 rows x 7 columns
# print(test)
# cols_list=df.columns[0:-1]
# target = df.columns[-1]
# # 数据标签
# cols_list = ["2","48"]
# # cols_list=df.columns[3:6]
# # target = df.columns[3]
# print(cols_list)
# target = "1"
# # 数据标签对应的数据集
x_train = train[cols_list]
# print(x_train)
y_train = train[target]
x_test = test[cols_list]
# print(x_test)
y_test = test[target]
#
x_train, x_test, y_train, y_test = train_test_split(df[cols_list],df[target], test_size=0.2,    #70%是训练集，30%测试集
                                                        random_state=123456)        #一样的随机种子

# x_tran,x_test,y_train,y_test = train_test_split(x, y, test_size=0.25)
"""开始训练"""
from sklearn.model_selection import GridSearchCV

parameters = {'C': [1,5,10,100,1000],
              'kernel': ['rbf','poly','linear'],
              }

model = SVR(C=10,kernel='rbf')
gs = GridSearchCV(estimator=model, param_grid=parameters, cv=5,
                  refit=True, verbose=2,n_jobs=-1,scoring='neg_mean_squared_error')  # param_grid为要调整参数的范围，cv为交叉验证的次数
# estimator为设置模型，scoring表用什么准则来评价
# print(gs)#查看gs中的所有参数
print(x_train.shape)
print(y_train.shape)
model.fit(x_train, y_train)
gs.fit(x_train, y_train)

print('最优参数: ', gs.best_params_)
predict_y = gs.predict(x_test)  # 使用找到的最佳参数对估计器进行调用预测


print('最优参数: ', gs.best_params_)

# clf.fit(x_train, y_train)
# predict_y = clf.predict(x_test)
# print("得分:", r2_score(y_test, y_hat))

"""计算预测值和真实值的均方根误差和绝对百分比标准误差"""
y_test=np.array(min_max_scaler.inverse_transform([y_test]))
predict_y=min_max_scaler.inverse_transform([predict_y])
y_test=y_test[0,:]
predict_y=predict_y[0,:]

rmse = math.sqrt(mean_squared_error(y_test, predict_y))  # 计算均方根误差
mse = mean_squared_error(y_test,predict_y)  # 计算均方误差
mae = mean_absolute_error(y_test, predict_y)  # 计算平均绝对误差
mape = get_mape(y_test, predict_y)
accuracy=1-mape

print('RMSE:%.14f' % rmse)
print('MSE:%.14f' % mse)
print('MAE:%.14f' % mae)
print('MAPE:%.14f' % mape)
# print('accuracy:%.4f' % accuracy)
print("R2=",r2_score(y_test,predict_y))
print("model_use=svr_",model)
#
# df1 = pd.DataFrame({'test': y_test, 'pred': predict_y})
# df1.to_excel('./datasave/{}.xlsx'.format(path[:-4]),index=True,header=True,index_label=True)
# print(df1)
# df1.to_csv('data5/Result{}.csv'.format(2), index=True, header=True, index_label=None)
# print(y_test)

plt.plot(range(len(y_test)), predict_y,"o", color='blue', label="predict")
plt.plot(range(len(y_test)), y_test ,color='black', label="real")
plt.legend()
plt.show()