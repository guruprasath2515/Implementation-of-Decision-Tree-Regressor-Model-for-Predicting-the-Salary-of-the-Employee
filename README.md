# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import pandas

2.Import Decision tree classifier

3.Fit the data in the model

4.Find the accuracy score


## Program:


Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

Developed by: GURU PRASATH R

RegisterNumber: 212223040053


```
import pandas as pd
data=pd.read_csv(r"E:\Desktop\CSE\Introduction To Machine Learning\dataset\Salary.csv")
print(data.head())
print(data.info())
print(data.isnull().sum())

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
print(data.head())

x=data[["Position","Level"]]
print(x.head())
y=data["Salary"]
print(y.head())

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
print(y_pred)
from sklearn.metrics import r2_score
r2=r2_score(y_test,y_pred)
print(r2)

pre=dt.predict([[5,6]])
print(pre)
```

## Output:

![11](https://github.com/user-attachments/assets/150033c1-b098-4e70-acb2-17724131021a)

![22](https://github.com/user-attachments/assets/32a38f18-2de2-4a6b-ab5d-e6dc64976799)

![33](https://github.com/user-attachments/assets/1372ae8f-d529-490a-b4e8-020e1944497a)

![44](https://github.com/user-attachments/assets/19f28956-3799-488b-8df0-5bd061c849b9)

![55](https://github.com/user-attachments/assets/32a6e195-b762-4e52-a0b2-baa4714dc91f)

![66](https://github.com/user-attachments/assets/7c76dc0f-6a01-4661-b4a6-a9012904cce1)

![77](https://github.com/user-attachments/assets/0ea0c0dc-b41d-49c1-b1bf-5a406313c48b)

![88](https://github.com/user-attachments/assets/86ba6aa6-d608-41a2-8efc-71339fce7205)

![99](https://github.com/user-attachments/assets/fc5fb41d-72aa-447b-b74f-bf571798cab2)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
