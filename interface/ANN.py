# -*- coding: utf-8 -*-
# @Time    : 2022/06/11 15:47
# @Author  : Deng Jiandong

from sklearn.neural_network import MLPRegressor
import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score#R square
import os
# 解决图表中汉字无法显示问题
from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

"""计算平均绝对百分误差 (MAPE)"""

def get_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))


test_size = 0.2  # 用作测试集的数据比例
model_seed = 100

path = 'surface tension+RDF.xlsx'

#读取文件
df = pd.read_excel(path, encoding='utf-8')

# 数据标签
cols_list=df.columns[1:-1]

# cols_list = ['T','A','B']
# target = df.columns[3]
# print(cols_list)
# 数据标签对应的数据集要预测的列：
target = df.columns[-1]

from sklearn.model_selection import train_test_split
import sklearn.preprocessing
min_max_scaler = sklearn.preprocessing.MinMaxScaler()
for i in range(len(cols_list)):
    df[cols_list[i]] = min_max_scaler.fit_transform(df[cols_list[i]].values.reshape(-1, 1))
df[target] = min_max_scaler.fit_transform(df[target].values.reshape(-1, 1))
x_train, x_test, y_train, y_test = train_test_split(df[cols_list],df[target], test_size=0.2,    #70%是训练集，30%测试集
                                                    random_state=123456)        #一样的随机种子



from sklearn.model_selection import GridSearchCV

parameters = {'hidden_layer_sizes': [1,5,10,20],
              'activation': ['logistic', 'relu', 'softmax', 'tanh'],
              'solver': ['sgd', 'adam', 'lbfgs'],
              'random_state':[123456]
              }
ann_model = MLPRegressor(hidden_layer_sizes=[20], activation='relu', solver='lbfgs', random_state=123456)



gs = GridSearchCV(estimator=ann_model, param_grid=parameters, cv=5,
                  refit=True, verbose=2,n_jobs=-1,scoring='neg_mean_squared_error')  # param_grid为要调整参数的范围，cv为交叉验证的次数
# estimator为设置模型，scoring表用什么准则来评价
# print(gs)#查看gs中的所有参数

print(x_train.shape)
print(y_train.shape)
ann_model.fit(x_train, y_train)
print(x_train)
print(y_train)

gs.fit(x_train, y_train)
print(gs.best_params_)

# exit()
predict_y=ann_model.predict(x_test)
"""反归一化"""
y_test=np.array(min_max_scaler.inverse_transform([y_test]))
predict_y=min_max_scaler.inverse_transform([predict_y])
y_test=y_test[0,:]
predict_y=predict_y[0,:]
mape = get_mape(y_test, predict_y)
rmse = np.sqrt(mean_squared_error(y_test, predict_y))  # 计算均方根误差
mse = mean_squared_error(y_test,predict_y)  # 计算均方误差
mae = mean_absolute_error(y_test, predict_y)  # 计算平均绝对误差

print('RMSE:%.15f' % rmse)
print('MSE:%.15f' % mse)
print('MAE:%.15f' % mae)
print('MAPE:%.15f' % mape)

print("R2=",r2_score(y_test,predict_y))
df2=pd.DataFrame({"test":y_test,'prediction':predict_y})
df2.to_csv('result{}.csv'.format(path[:-5]),index=False)