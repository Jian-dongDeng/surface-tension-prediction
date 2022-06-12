#coding:utf8
import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score#R square
from xgboost import XGBRegressor
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

path = 'surface tension+interpret+RDF.xlsx'


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

# 数据标签
cols_list=df.columns[1:-1]

# cols_list = ['T','A','B']
# target = df.columns[3]
# print(cols_list)
# 数据标签对应的数据集要预测的列：
target = df.columns[-1]
# target = 'C'
# 数据标签对应的数据集


from sklearn.model_selection import train_test_split
# import sklearn.preprocessing
# min_max_scaler = sklearn.preprocessing.MinMaxScaler()
# for i in range(len(cols_list)):
#     df[cols_list[i]] = min_max_scaler.fit_transform(df[cols_list[i]].values.reshape(-1, 1))
# df[target] = min_max_scaler.fit_transform(df[target].values.reshape(-1, 1))
x_train, x_test, y_train, y_test = train_test_split(df[cols_list],df[target], test_size=0.2,    #70%是训练集，30%测试集
                                                    random_state=123456)        #一样的随机种子

"""开始训练"""
from sklearn.model_selection import GridSearchCV

parameters = {'n_estimators': [500,1000,2000],
              'max_depth': [2,4,7,10],
              'learning_rate': [0.03,0.05,0.1],
              }

model = XGBRegressor(seed=12345,  # 随机种子
                     n_estimators=2000,  # 弱分类器数目，缺省：100
                     max_depth=10,  # 树的最大深度，树越深通常模型越复杂，更容易过拟合
                     learning_rate=0.05)  # 节点分裂所需要的最小损失函数下降值
gs = GridSearchCV(estimator=model, param_grid=parameters, cv=5,
                  refit=True, verbose=2,n_jobs=-1,scoring='neg_mean_squared_error')  # param_grid为要调整参数的范围，cv为交叉验证的次数
# estimator为设置模型，scoring表用什么准则来评价
# print(gs)#查看gs中的所有参数

print(x_train.shape)
print(y_train.shape)
model.fit(x_train, y_train)
# gs.fit(x_train, y_train)
# print(gs.best_params_)
print(model.feature_importances_)
# exit()
predict_y=model.predict(x_test)
# exit()
# gs.fit(x_train, y_train)
# predict_y=gs.predict(x_test)
import shap
# print(model.feature_importances_)
shap.initjs()  # notebook环境下，加载用于可视化的JS代码

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(x_train,y_train)  # 传入特征矩阵X，计算SHAP值
#shap.force_plot(explainer.expected_value, shap_values[0,:], x_train.iloc[0,:],matplotlib=True)

# summarize the effects of all the features
shap.summary_plot(shap_values, x_train)

shap.summary_plot(shap_values, x_train, plot_type="bar")

# shap_interaction_values = explainer.shap_interaction_values(x_train)
# shap.summary_plot(shap_interaction_values, x_train)

shap.dependence_plot("t", shap_values, x_train,interaction_index="mass")

# print(model.feature_importances_)
# print('最优参数: ', gs.best_params_)
# predict_y = gs.predict(x_test)  # 使用找到的最佳参数对估计器进行调用预测
#
# print('最优参数: ', gs.best_params_)


# from sklearn.inspection.partial_dependence import partial_dependence
# pdp,anxs=partial_dependence(model_svr,x,[0],grid_resolution=150)
#
# #plt.plot(range(len(feature_test)), y_valid)  # 测试数组
#
# # plt.plot(anxs, pdp,'o',color='blue')  # 测试数组
# print(len(pdp[0]),len(anxs[0]))
# df3 = pd.DataFrame({'pdp': pdp[0], '5': anxs[0]})
# df3.to_excel('interpret0.xlsx', index=True)
# plt.show()

# fig = plt.figure()
# # target_feature = (1, 3)
#
# pdp, axes = partial_dependence(model, x_test,[0,3], grid_resolution=118)
#
# df1=pd.DataFrame(pdp[0])
# # df1.to_csv('+δq.csv')
# XX, YY = np.meshgrid(axes[0], axes[1])
# Z = pdp[0].reshape(list(map(np.size, axes))).T
#
# from mpl_toolkits.mplot3d import Axes3D
# ax = Axes3D(fig)
# surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1, cmap=plt.cm.BuPu, edgecolor='k')
#
# ax.set_xlabel('T/K')
# ax.set_ylabel('')
# ax.set_zlabel('Partial dependence')
#
# #  pretty init view
# ax.view_init(elev=42, azim=115)
# plt.colorbar(surf)
# plt.subplots_adjust(top=0.8)
# # plt.savefig('./fig/+α+δq.png')
# plt.show()





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
# plt.plot(range(len(predict_y)), predict_y, color='blue', label="predict")
# plt.plot(range(len(y_test)), y_test, color='black', label="real")
# plt.legend()
# plt.show()