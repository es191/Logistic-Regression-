# Import the necessary libraries
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# Reading our file
ad_data = pd.read_csv('advertising.csv')
ad_data.head()
ad_data.describe()
ad_data.info()

#Creating a histogram of the 'Age' 
sns.set_style('ticks')
ad_data['Age'].hist(bins=30)
plt.xlabel('Age')

#Creating a jointplot showing 'Area Income' versus 'Age'
sns.jointplot(x='Age', y='Area Income', data = ad_data)

#Creating a jointplot showing the kde distributions of 'Daily Time spent on site' vs. 'Age'
sns.jointplot(x='Age',y='Daily Time Spent on Site',data=ad_data,color='red',kind='kde')

#Creating a jointplot of 'Daily Time Spent on Site' vs. 'Daily Internet Usage'
sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',data=ad_data, color='green')

#Creating a pairplot with the hue = 'Clicked on Ad'
sns.pairplot(ad_data, hue='Clicked on Ad')\

#Spliting our data to train and test data
from sklearn.model_selection import train_test_split
X = ad_data[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','Male']]
y = ad_data['Clicked on Ad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#Traning and fitting our model
from sklearn.linear_model import LogisticRegression
LogReg = LogisticRegression()
LogReg.fit(X_train,y_train)

#Predictions of out model
predictions = LogReg.predict(X_test)

#Classification report for our model
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
