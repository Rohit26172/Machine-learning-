# Machine-learning-
# this is the import the library 

import numpy as np
import pandas as pd
import seaborn as lineplot 
from matplotlib import pyplot 

url ="https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-car-sales.csv"
dataset = pd.read_csv(url)
# show the only ten values 
dataset.head(10)
# all data the numerical values and statical infromatiom do
dataset.describe()

# the info is use by only all summary see
dataset.info()

# unique data
load_dataset = pd.DataFrame()
load_dataset["Sales"]= dataset["Sales"]
load_dataset["Month"] = dataset["Month"]

# use the linearEncoder 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
# fit the dataset and there are training and testing that dataset 
dataset["Sales"] =le.fit_transform(dataset["Sales"])
dataset["Sales"].unique()

# use the iloc 
dataset.iloc[:,0] =le.fit_transform(dataset.iloc[:,0])
# prefrom the iloc 
x = dataset.iloc[:,:-1]
y = dataset.iloc[:,:-1]

# SPLITING DATA SET INTO TRAINING AND TESTING DATA SETS
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x ,y ,train_size =0.2 ,random_state = 2)
# linear model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
# predicted dataset
y_pred = regressor.predict(x_test)

pd.DataFrame(
{
         'ACTUAL':y_test,
         'PREDICTED':y_pred
     }
   ).head()
