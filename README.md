# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:

```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:A.Sesank
RegisterNumber:23009543
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')

# display the content in datafield
df.head()
df.tail()

#Segregating data to variables

x = df.iloc[:,:-1].values
print(x)


y = df.iloc[:,1].values
print(y)

#Splitting train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)

#displaying predicted values
print(y_pred)

#displaying actual values
print(y_test)

#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)

Displaying the content in datafield

head:

![417421853-26da9207-078b-47e1-a65a-5567b58961ec](https://github.com/user-attachments/assets/89a804d4-5193-4d43-96b7-f726a0c3bb01)

tail:

![417422158-e9c2e062-83ea-4b85-bce9-59169a8f92fb](https://github.com/user-attachments/assets/bb853c44-3789-4215-a733-f02680a92506)

Segregating data to variables:


![417422369-42c9efdb-a110-4f02-afa2-c166d35c5329](https://github.com/user-attachments/assets/d623aafc-2620-4f09-abcd-071da491c323)

![417422608-5b8ca5a5-f11b-4b94-98cb-2584234dcf94](https://github.com/user-attachments/assets/64d7043b-744b-4dcd-ba1d-30498b7a29f8)

Displaying predicted values:

![417422961-95038c76-689d-4e86-94d4-06c7b706f0ea](https://github.com/user-attachments/assets/fd622d74-cd9d-468d-a320-13cb36846a4e)

Displaying actual values:

![417423265-1a61dd1a-2f23-429b-a865-69b6a9be18a0](https://github.com/user-attachments/assets/d45799cc-e0d3-4b9e-bd3e-cdd0d2ea94b4)

Graph plot for training data:

![417423429-e3132461-e5d0-499d-a866-0d53b5298abb](https://github.com/user-attachments/assets/219f05e5-94ac-4816-9612-ca5a5593b97d)

Graph plot for test data:

![417423575-0fef34ce-2e74-4132-8203-1fd79322161f](https://github.com/user-attachments/assets/5c663ea0-c10e-469b-ba9c-cbd73ff3be2c)

MSE MAE RMSE:

![417423811-5dbe3c3d-02f3-4329-bcaf-647c5c95403e](https://github.com/user-attachments/assets/c8600ed6-9de9-42e3-95d2-fc92c35b6a94)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
