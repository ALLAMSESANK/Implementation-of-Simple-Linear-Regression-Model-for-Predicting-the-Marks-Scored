# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored
## Date : 8/03/25
## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries and read the dataframe.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas. 



## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:A.Sesank
RegisterNumber:  212223240006
*/


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')

df.head()

df.tail()

#Array value of X
X=df.iloc[:,:-1].values
X

#Array value of Y
Y=df.iloc[:,1].values
Y

#splitting train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

#displaying predicted values
Y_pred

#displaying actual values
Y_test

#graph plot for training data
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Hours Vs Scores(Train Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#graph plot for test data
plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours Vs Scores(Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```

## Output:
### Head Values
![image](https://github.com/user-attachments/assets/f8383e2e-1bcc-4137-b1cd-fd706bd438fb)


### Tail Values

![image](https://github.com/user-attachments/assets/d3b67821-e803-4413-9f0f-415f6b04a21f)

### X Values
![image](https://github.com/user-attachments/assets/a4a3710b-6d70-4286-a728-cc641ba21638)


### y Values


### Predicted Values

![image](https://github.com/user-attachments/assets/eee55077-1f9e-4a60-9471-d5e2a131a6e8)

### Actual Values
![image](https://github.com/user-attachments/assets/0b4bed37-2dea-4330-abcc-cfb278976f69)


### Training Set
![image](https://github.com/user-attachments/assets/d8a0d3b8-d698-4bbf-a7d1-505f963a0734)


### Testing Set
![image](https://github.com/user-attachments/assets/83dc9c01-c442-4aaf-ad30-245eb1905a07)


### MSE, MAE and RMSE
![image](https://github.com/user-attachments/assets/c438fe10-89be-4e8c-9eda-0e436fbf83ca)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

