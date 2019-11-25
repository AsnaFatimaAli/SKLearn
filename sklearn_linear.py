from sklearn import datasets
from sklearn.linear_model import LinearRegression
import pandas as pd

boston_dataset = datasets.load_boston() # to load the boston data set
boston = pd.DataFrame(boston_dataset.data) # create a data frame of the data
boston.columns = boston_dataset.feature_names # rename the columns 
boston['House_Price' ]= boston_dataset.target # add in the y values into the data frame
x = boston.drop(["House_Price", "CHAS"], axis =1) # create x for regression without the y value and the feature with dummy variables
y = boston['House_Price'] # create the y variable for regression
lr = LinearRegression().fit(x,y) # create a linear regression instance and fit x and y 
coefficients = lr.coef_ # get the coefficients
coefficients = [abs(x) for x in coefficients ] # get absolute value of the coefficients to take into account most pos and neg

def effective(ls):
    ls = max(ls)
    for index, num in enumerate(coefficients):
        if num == ls:
            return x.columns[index]

effective(coefficients)

# NOX is the most effective







