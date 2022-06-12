#coding:utf8
import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score#R square
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
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
# 数据集划分
# print(df.shape)
# num_test = int(test_size * len(df))
# num_train = len(df) - num_test
# print(num_train)
# train = df[:num_train]  # 959 rows x 7 columns
# print(train)
# test = df[num_train:]  # 239 rows x 7 columns
# print(test)
#
# # 数据标签
# cols_list = ["2","48"]
# # cols_list=df.columns[3:6]
# # target = df.columns[3]
# print(cols_list)
# target = "1"
# # 数据标签对应的数据集
# x_train = train[cols_list]
# print(x_train)
# y_train = train[target]
# x_test = test[cols_list]
# print(x_test)
# y_test = test[target]

cols_list=df.columns[1:-1]
target = df.columns[-1]

'''归一化'''
# import sklearn.preprocessing
# min_max_scaler = sklearn.preprocessing.MinMaxScaler()
# for i in range(len(cols_list)):
#     df[cols_list[i]] = min_max_scaler.fit_transform(df[cols_list[i]].values.reshape(-1, 1))
# df[target] = min_max_scaler.fit_transform(df[target].values.reshape(-1, 1))
'''随机划分数据集'''
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df[cols_list],df[target], test_size=0.2,    #70%是训练集，30%测试集
                                                        random_state=123456)        #一样的随机种子

"""开始训练"""
from sklearn.model_selection import GridSearchCV
# parameters = {'n_estimators': range(20, 120, 5),
#               'max_depth': range(2, 6, 1),
#               'learning_rate': [0.1],
#               'min_child_weight': [20],  # 范围为5-21，步长为1
#               }

# model = XGBRegressor(seed=model_seed,  # 随机种子
#                      n_estimators=115,  # 弱分类器数目，缺省：100
#                      max_depth=5,  # 树的最大深度，树越深通常模型越复杂，更容易过拟合
#                      eval_metric='rmse',
#                      learning_rate=0.1,
#                      min_child_weight=4,  # 叶子节点需要的最小样本权重和
#                      subsample=0.8,  # 构造每棵树的所用样本比例（样本采样比例）
#                      colsample_bytree=0.8,  # 构造每棵树的所用特征比例
#                      colsample_bylevel=0.8,  # 树在每层每个分裂的所用特征比例0.7，通常在0.7-0.8可以得到较好的结果
#                      gamma=0)  # 节点分裂所需要的最小损失函数下降值
# gs = GridSearchCV(estimator=model, param_grid=parameters, cv=5,
#                   refit=True, scoring='neg_mean_squared_error')  # param_grid为要调整参数的范围，cv为交叉验证的次数
# estimator为设置模型，scoring表用什么准则来评价
# print(gs)#查看gs中的所有参数20
print(x_train.shape)
print(y_train.shape)
parameters = {'n_estimators': [500,1000,2000],

              'learning_rate': [0.03,0.05,0.1],
              }





bdt = AdaBoostRegressor(DecisionTreeRegressor(max_depth=7),learning_rate=0.05,n_estimators=2000
                        )
gs = GridSearchCV(estimator=bdt, param_grid=parameters, cv=5,
                  refit=True, verbose=2,n_jobs=-1,scoring='neg_mean_squared_error')  # param_grid为要调整参数的范围，cv为交叉验证的次数
# estimator为设置模型，scoring表用什么准则来评价
# print(gs)#查看gs中的所有参数

print(x_train.shape)
print(y_train.shape)
bdt.fit(x_train, y_train)
# gs.fit(x_train, y_train)
# print(gs.best_params_)
predict_y=bdt.predict(x_test)
# exit()

print(bdt.feature_importances_)
# print('最优参数: ', gs.best_params_)
predict_y = bdt.predict(x_test)


# clf.fit(x_train, y_train)
# predict_y = clf.predict(x_test)
# print("得分:", r2_score(y_test, y_hat))
'''反归一化'''
# y_test=np.array(min_max_scaler.inverse_transform([y_test]))
# predict_y=min_max_scaler.inverse_transform([predict_y])
# y_test=y_test[0,:]
# predict_y=predict_y[0,:]

"""计算预测值和真实值的均方根误差和绝对百分比标准误差"""
RMSE = math.sqrt(mean_squared_error(y_test, predict_y))  # 计算均方根误差
MSE = mean_squared_error(y_test,predict_y)  # 计算均方误差
MAE = mean_absolute_error(y_test, predict_y)  # 计算平均绝对误差
MAPE = get_mape(y_test, predict_y)
R2=r2_score(y_test,predict_y)
accuracy=1-MAPE

print('RMSE:%.15f' % RMSE)
print('MSE:%.15f' % MSE)
print('MAE:%.15f' % MAE)
print('MAPE:%.15f' % MAPE)
# print('accuracy:%.15f' % accuracy)
print("R2=",r2_score(y_test,predict_y))

"""绘制真实值和预测值的折线图"""
import matplotlib.pyplot as plt
df1=pd.DataFrame({'test':y_test,'prediction':predict_y})

# print(df1)
plt.plot(range(len(df1['test'])),df1['test'], color='blue', label='Target')
plt.plot(range(len(df1['prediction'])),df1['prediction'], 'o',color='black', label='Prediction')
plt.legend()



# save_path = r'E:\Python\{}.png'
# for m,n in zip(cols_list,model.feature_importances_):
#     print('{}:{:.2f}%'.format(m,n*100))
plt.show()