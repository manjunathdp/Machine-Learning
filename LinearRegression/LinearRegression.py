
from sklearn.metrics import mean_squared_error  # For finding error
from sklearn.linear_model import LinearRegression  # For leanear regression model
# For spliting data into test and train
from sklearn.model_selection import train_test_split
import pandas as pd  # For reading csv files
Boston = pd.read_csv('Boston.csv')
BostonHead = Boston.head()
print(BostonHead)
print(Boston.columns)
# here 'y' is the dependent veriable
y = Boston[['medv']]
# here 'x' is the independent veriable
x = Boston[['crim']]
# train_test_split takes 3 parameters (1st independent 2nd dependent 3rd test size)
# it will give you 4 results so we should store them in 4 different veriables
# x_train,x_test,y_train,y_test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
LR_instance = LinearRegression()  # LinearRegression instance will be created
LR_instance.fit(x_train, y_train)  # Fitting the model on Training data
# Predicting the values using test data and the predicted values will be stored in predit
predict = LR_instance.predict(x_test)
# comparing Actual values and predicted values
print(y_test.head())  # Actual values
print(predict[0:5])  # Predicted values
# calculating Residual that is difference between Actual values and Predicted values
print("Error:", mean_squared_error(y_test, predict))


# ________Model 2___________

# Chenging the indipendent veriable and then predicting the values
# Notice the Error

x = Boston[['lstat']]
y = Boston[['medv']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
LR_instance = LinearRegression()  # LinearRegression instance will be cr
LR_instance.fit(x_train, y_train)  # Fitting the model on Training data
predict = LR_instance.predict(x_test)
print(y_test.head())  # Actual values
print(predict[0:5])  # Predicted values
print("Error:", mean_squared_error(y_test, predict))

# ________Model 3___________

# Chenging the indipendent veriable and then predicting the values with multipal features
# Notice the Error

# 'dis'=35  'nox'=34  'rm'=31 'rad'=35 'tax'=33 'ptratio'=28
x = Boston[['ptratio', 'rm']]
y = Boston[['medv']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
LR_instance = LinearRegression()  # LinearRegression instance will be cr
LR_instance.fit(x_train, y_train)  # Fitting the model on Training data
predict = LR_instance.predict(x_test)
print(y_test.head())  # Actual values
print(predict[0:5])  # Predicted values
print("Error:", mean_squared_error(y_test, predict))
