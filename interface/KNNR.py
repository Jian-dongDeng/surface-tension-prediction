from sklearn.neighbors import KNeighborsRegressor

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
#选择模型；'lin','rbf','poly'

test_size = 0.2  # 用作测试集的数据比例
model_seed = 42

path = 'surface tension+RDF.xlsx'

#读取文件
df = pd.read_excel(path, encoding='utf-8')
# 数据集划分

cols_list=df.columns[1:-1]
target = df.columns[-1]
import sklearn.preprocessing
min_max_scaler = sklearn.preprocessing.MinMaxScaler()
for i in range(len(cols_list)):
    df[cols_list[i]] = min_max_scaler.fit_transform(df[cols_list[i]].values.reshape(-1, 1))
df[target] = min_max_scaler.fit_transform(df[target].values.reshape(-1, 1))


x_train, x_test, y_train, y_test = train_test_split(df[cols_list],df[target], test_size=0.2,    #70%是训练集，30%测试集
                                                        random_state=123456)


# clf = SVR(kernel='linear', C=1.25)
model=KNeighborsRegressor(n_neighbors=3, weights='uniform',
                           algorithm='auto', leaf_size=10,
                           p=2, metric='minkowski', metric_params=None,
                           n_jobs=-1)

from sklearn.model_selection import GridSearchCV
parameters = {'n_neighbors': [2,5,10,20],
              'algorithm': ['ball_tree','kd_tree','brute'],
              }
model=KNeighborsRegressor()

gs = GridSearchCV(estimator=model, param_grid=parameters, cv=5,
                   scoring='neg_mean_squared_error')  # param_grid为要调整参数的范围，cv为交叉验证的次数

gs.fit(x_train, y_train)
print(gs.best_params_)
predict_y = gs.predict(x_test)


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
print('accuracy:%.14f' % accuracy)
print("R2=",r2_score(y_test,predict_y))
print("model_use=",model)


# print(y_test)

plt.plot(range(len(y_test)), predict_y,'o', color='blue', label="predict")
plt.plot(range(len(y_test)), y_test, color='black', label="real")
plt.legend()
plt.show()