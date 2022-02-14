import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge,Lasso
data=pd.read_csv("insurance.csv",index_col=False)
# print(data.head())
data.replace({"sex":{"male":1,"female":0}},inplace=True)
data.replace({"smoker":{"yes":1,"no":0}},inplace=True)
data.drop("region",axis=1,inplace=True)
data.drop("children",axis=1,inplace=True)
# print(data.head())
# print(data.describe())
# plt.bar(data["sex"],data['region'])
# plt.show()

x = data.drop('charges', axis=1)
y = data[['charges']]
# # print(x.shape)
# # print(y.shape)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=51)
# # print(x_train)
# print(y_train)
# print(x_test)
# print(y_test)
lr=LinearRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_train)
# print(y_pred)
# print(y_test)
# print(lr.score(x_test,y_test))
# rd=Ridge()
# rd.fit(x_train,y_train)
# print(rd.score(x_test,y_test))
# ls=Lasso()
# ls.fit(x_train,y_train)
# print(ls.score(x_test,y_test))
import joblib
joblib.dump(lr,"insurance_cost_prediction_model.pkl")
input=(19,1,27.9,0)
input2=np.asarray(input)
# input3=np.reshape(-1,1)
prabhat=lr.predict([input2])[0][0]
print(prabhat)