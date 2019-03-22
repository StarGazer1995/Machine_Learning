import plt
import decisiontree as deci
import bayes as bayes
import linear_regression as linear
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as pp

x,y=linear.createFunction()
X_train,X_test,Y_train,Y_test=train_test_split(x,y,random_state=0)
linreg=LinearRegression().fit(X_train,Y_train)
estimated_linearmodel=linreg.intercept_+linreg.coef_*x
pp.figure(figsize=(5,4))
pp.scatter(x,y,marker='o')
pp.plot(x,estimated_linearmodel,'b-')
pp.show()