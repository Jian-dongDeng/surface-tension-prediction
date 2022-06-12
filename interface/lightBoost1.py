#coding:utf8
import math
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMRegressor
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score#R square
# from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import os
# 解决图表中汉字无法显示问题
from pylab import *
from sklearn.model_selection import train_test_split
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

"""计算平均绝对百分误差 (MAPE)"""

def get_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))


test_size = 0.2  # 用作测试集的数据比例
model_seed = 100
stock_title='1'

path = 'surface tension+interpret+RDF.xlsx'

df = pd.read_excel(path, encoding='gbk')
# 数据集划分
# cols_list = ["2","48"]
#
# cols_list=df.columns[0:9]
# print(cols_list)
# target = "1"
# y = df[target]
# X = df[cols_list]
# cols_list = ['CON','T','Y','Theta','Z','Yr','Yi','i/tan(The)','C','Er','Ei','Er/Ei']
# target = df.columns[10]
# print(cols_list)
# 数据标签对应的数据集要预测的列：
# target = 'S'
cols_list=df.columns[1:-1]
print(cols_list)
target = df.columns[-1]
# import sklearn.preprocessing
# min_max_scaler = sklearn.preprocessing.MinMaxScaler()
# for i in range(len(cols_list)):
#     df[cols_list[i]] = min_max_scaler.fit_transform(df[cols_list[i]].values.reshape(-1, 1))
# df[target] = min_max_scaler.fit_transform(df[target].values.reshape(-1, 1))
x_train, x_test, y_train, y_test = train_test_split(df[cols_list],df[target], test_size=0.2,    #70%是训练集，30%测试集
                                                        random_state=123456)

"""开始训练"""
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators': [500,1000,2000],
              'max_depth': [2,4,7,10],
              'learning_rate': [0.03,0.05,0.1],
              }

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
# model = CatBoostRegressor(
#         iterations=10000, learning_rate=0.03,
#         depth=3, l2_leaf_reg=3,
#         loss_function='MAE',
#         eval_metric='MAE',
#         random_seed=model_seed)
model = LGBMRegressor(objective = 'regression',

                        learning_rate = 0.1,
                        n_estimators = 2000,
                        max_depth = 4)
gs = GridSearchCV(estimator=model, param_grid=parameters, cv=5,
                  refit=True, verbose=2,n_jobs=-1,scoring='neg_mean_squared_error')  # param_grid为要调整参数的范围，cv为交叉验证的次数
# gs.fit(x_train,y_train)
# print('Best parameters found by rid search are:', gs.best_params_)
# exit()
# model=CatBoostRegressor(iterations=10000, loss_function='MAE', boosting_type='Ordered')
# gs = GridSearchCV(estimator=model, param_grid=parameters, cv=5,
#                   refit=True, scoring='neg_mean_squared_error')  # param_grid为要调整参数的范围，cv为交叉验证的次数
# # estimator为设置模型，scoring表用什么准则来评价
# # print(gs)#查看gs中的所有参数
print(x_train.shape)
print(y_train.shape)

model.fit(x_train, y_train)
# print('最优参数: ', gs.best_params_)
predict_y = model.predict(x_test)  # 使用找到的最佳参数对估计器进行调用预测
  # 将预测值添加到测试集中
# print(predict_y)
print(model.feature_importances_)
# print('--------------')
# print(test['predict_y'])
"""反归一化"""
# y_test=np.array(min_max_scaler.inverse_transform([y_test]))
# predict_y=min_max_scaler.inverse_transform([predict_y])
# y_test=y_test[0,:]
# predict_y=predict_y[0,:]
"""计算预测值和真实值的均方根误差和绝对百分比标准误差"""
rmse = math.sqrt(mean_squared_error(y_test, predict_y))  # 计算均方根误差
mse = mean_squared_error(y_test, predict_y)  # 计算均方误差
mae = mean_absolute_error(y_test, predict_y)  # 计算平均绝对误差
mape = get_mape(y_test, predict_y)
accuracy=1-mape

print('RMSE:%.14f' % rmse)
print('MSE:%.14f' % mse)
print('MAE:%.14f' % mae)
print('MAPE:%.14f' % mape)
print('accuracy:%.14f' % accuracy)
print("R2=",r2_score(y_test,predict_y))

"""绘制真实值和预测值的折线图"""
import matplotlib.pyplot as plt
k=cols_list
df1=pd.DataFrame({'test':y_test,'pred':predict_y})
# print(df1)

# ax = test.plot(title=stock_title, x=len(x_test), y=y_test, style='b-', grid=False)
plt.plot(range(len(x_test)), y_test, color="red",label='test')
plt.plot(range(len(x_test)), predict_y, color="blue",label='pred')
plt.legend()
# ax = test.plot(x=len(x_test), y=predict_y, style='r-', grid=False,ax=ax)
# left, width = .25, .5
# bottom, height = .25, .5
# right = left + width
# top = bottom + height
# ax2.text(0.6 * (left + right), 0.8 * (bottom + top),
#         ('test_size:%s' % test_size, ('MAPE:%.4f:' % mape), ('RMSE:%.4f:' % rmse)),
#         horizontalalignment='center',
#         verticalalignment='center',
#         fontsize=10, color='red',
#         transform=ax.transAxes)
# save_path = r''
# save_path = save_path.format(k)
# plt.savefig(save_path)
plt.show()
