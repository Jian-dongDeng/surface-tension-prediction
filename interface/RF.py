from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import r2_score

# np.random.seed(0)

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
cols_list=df.columns[1:-1]
target = df.columns[-1]
# trian_num=int(len(df)*(1-test_size))
#
#
#
# x_train = df.iloc[:trian_num,:-1]
# y_train = df.iloc[:trian_num,-1]
# x_test =  df.iloc[trian_num:,:-1]
# y_test = df.iloc[trian_num:,-1]


# print(x_train)
# print(y_train)
# print(x_test)

# cols_list = ["2","48"]
#
# # cols_list=df.columns[4:51]
# target = "1"
# y = df[target]
# X = df[cols_list]
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
# print(df.shape)
# num_test = int(test_size * len(df))
# num_train = len(df) - num_test
# # # print(num_train)
# train = df[:num_train]  # 959 rows x 7 columns
# # print(train)
# test = df[num_train:]  # 239 rows x 7 columns
# # # print(test)
# #
# # # 数据标签
# #
# # # cols_list=df.columns[1:3]
# #
# # # 数据标签对应的数据集
# x_train = train[cols_list]
# # print(x_train)
# y_train = train[target]
# x_test = test[cols_list]
# # print(y_train)
# # print(x_test)
# y_test = test[target]

#读取文件
# 数据集划分
# cols_list = ["2","48"]
#
# cols_list=df.columns[0:9]
# print(cols_list)
# target = "1"
# y = df[target]
# X = df[cols_list]
# cols_list = ['CON','T','Y','Theta','Z','Yr','Yi','i/tan(The)','C','S','SS']
# target = df.columns[10]
# print(cols_list)
# 数据标签对应的数据集要预测的列：

# target = 'Er'
# import sklearn.preprocessing
# min_max_scaler = sklearn.preprocessing.MinMaxScaler()
# for i in range(len(cols_list)):
#     df[cols_list[i]] = min_max_scaler.fit_transform(df[cols_list[i]].values.reshape(-1, 1))
# df[target] = min_max_scaler.fit_transform(df[target].values.reshape(-1, 1))
x_train, x_test, y_train, y_test = train_test_split(df[cols_list],df[target], test_size=0.2,    #70%是训练集，30%测试集
                                                        random_state=123456)        #一样的随机种子

"""开始训练"""

# clf = SVR(kernel='linear', C=1.25)
model=RandomForestRegressor(n_estimators=100,
                            oob_score = True,
                            n_jobs = -1,
                            max_features = "log2",
                            max_depth=20)
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators': [100,500,1000,2000],
              'max_features': ['auto', 'sqrt', 'log2'],
              'max_depth':[1,5,10,20]
              }
# model.fit(x_train, y_train)
gs = GridSearchCV(estimator=model, param_grid=parameters, cv=5,
                  refit=True, verbose=2,n_jobs=-1,scoring='neg_mean_squared_error')  # param_grid为要调整参数的范围，cv为交叉验证的次数
# estimator为设置模型，scoring表用什么准则来评价
# print(gs)#查看gs中的所有参数

print(x_train.shape)
print(y_train.shape)
model.fit(x_train, y_train)
print(x_train)
print(y_train)

# gs.fit(x_train, y_train)
# print(gs.best_params_)

predict_y = model.predict(x_test)
print(model.feature_importances_)
# import shap
# print(model.feature_importances_)
# shap.initjs()  # notebook环境下，加载用于可视化的JS代码
#
# explainer = shap.TreeExplainer(model)
#
# shap_values = explainer.shap_values(x_train,y_train)  # 传入特征矩阵X，计算SHAP值
#
# #shap.force_plot(explainer.expected_value, shap_values[0,:], x_train.iloc[0,:],matplotlib=True)
#
# # summarize the effects of all the features
# shap.summary_plot(shap_values, x_train)
#
# shap.summary_plot(shap_values, x_train, plot_type="bar")
#
# # shap_interaction_values = explainer.shap_interaction_values(x_train)
# # shap.summary_plot(shap_interaction_values, x_train)
#
# shap.dependence_plot("t", shap_values, x_train,interaction_index="mass")
# clf.fit(x_train, y_train)
# predict_y = clf.predict(x_test)
# print("得分:", r2_score(y_test, y_hat))

"""计算预测值和真实值的均方根误差和绝对百分比标准误差"""
# y_test=np.array(min_max_scaler.inverse_transform([y_test]))
# predict_y=min_max_scaler.inverse_transform([predict_y])
# y_test=y_test[0,:]
# predict_y=predict_y[0,:]
RMSE = math.sqrt(mean_squared_error(y_test, predict_y))  # 计算均方根误差
MSE = mean_squared_error(y_test,predict_y)  # 计算均方误差
MAE = mean_absolute_error(y_test, predict_y)  # 计算平均绝对误差
MAPE = get_mape(y_test, predict_y)
R2=r2_score(y_test,predict_y)
accuracy=1-MAPE

print('RMSE:%.14f' % RMSE)
print('MSE:%.14f' % MSE)
print('MAE:%.14f' % MAE)
print('MAPE:%.14f' % MAPE)
# print('accuracy:%.14f' % accuracy)
print("R2=",r2_score(y_test,predict_y))

"""绘制真实值和预测值的折线图"""
import matplotlib.pyplot as plt
df1=pd.DataFrame({'test':y_test,'prediction':predict_y})
# print(df1)


plt.plot(range(len(df1['test'])),df1['test'], color='blue', label='Target')
plt.plot(range(len(df1['prediction'])),df1['prediction'], color='black', label='Prediction')
plt.legend()

# save_path = r'E:\Python\{}.png'
# for m,n in zip(cols_list,model.feature_importances_):
#     print('{}:{:.2f}%'.format(m,n*100))
plt.show()