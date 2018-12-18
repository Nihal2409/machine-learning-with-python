import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Get the Data

train = pd.read_csv('advertising.csv')
train.head()
train.info()
train.describe()

# Exploratory Data Analysis

sns.distplot(train['Age'],bins=30,kde=False)

sns.jointplot(x='Age',y='Area Income',data=train)
sns.jointplot(y='Daily Time Spent on Site',x='Age',data= train,kind='kde')
sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',data=train)

sns.pairplot(data=train,hue='Clicked on Ad')


# Logistic Regression

X=train[['Daily Time Spent on Site', 'Age', 'Area Income',
       'Daily Internet Usage', 'Male']]
y=train['Clicked on Ad']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()

lr.fit(X_train,y_train)

# Predictions and Evaluations

predictions=lr.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(predictions,y_test))