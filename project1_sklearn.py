#本答案只考虑了前8个小时PM2.5的影响

#导入所有需要用的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as LR 
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score 

#导入数据，由于数据列名有些乱码，所以需要重新设定一下
names = ['date','place','item',0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
df = pd.read_csv('./train.csv',names = names)
#去掉碍眼的无关数据，对字符进行替换
df.drop([10,11,12,13,14,15,16,17,18,19,20,21,22,23],axis = 1,inplace = True)
df.drop(index = 0,inplace = True)
df.replace('NR',0,inplace = True)

#设定自变量，因变量，改变因变量的数据形式
pm25 = df[df['item'] == 'PM2.5'].iloc[:,3:]
X = pm25.iloc[:,:9]
Y = pm25.iloc[:,9]
Y = np.array(Y,float)

#分训练集，训练集
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y,test_size=0.3,random_state=20)

#建立并训练线性回归模型
reg = LR().fit(Xtrain,Ytrain)
yhat = reg.predict(Xtest)
erro = abs(yhat-Ytest).sum()/len(Ytest)
print(erro)  #4.6
