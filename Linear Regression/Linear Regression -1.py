import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# Get the Data
# We'll work with the Ecommerce Customers csv file from the company. It has Customer info, such as Email, Address, and their color Avatar. Then it also has numerical value columns:
# * Avg. Session Length: Average session of in-store style advice sessions.
# * Time on App: Average time spent on App in minutes
# * Time on Website: Average time spent on Website in minutes
# * Length of Membership: How many years the customer has been a member. 

Customers=pd.read_csv('Ecommerce Customers')
Customers.head()
Customers.describe()
Customers.info()

# Exploratory Data Analysis
# Let's explore the data!

sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=Customers)
sns.jointplot(x='Time on App',y='Length of Membership',data=Customers,kind='hex')
sns.pairplot(Customers)

sns.lmplot(x='Yearly Amount Spent',y='Length of Membership',data=Customers)


# Training and Testing Data


X=Customers[['Avg. Session Length', 'Time on App',
       'Time on Website', 'Length of Membership']]

y=Customers['Yearly Amount Spent']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=101)

from sklearn.linear_model import LinearRegression
lm=LinearRegression()

lm.fit(X_train,y_train)
print(lm.coef_)

prediction =lm.predict(X_test)
sns.scatterplot(y_test,prediction)


# Evaluating the Model
from sklearn import metrics
print('MAE: ', metrics.mean_absolute_error(y_test,prediction))
print('MSE: ', metrics.mean_squared_error(y_test,prediction))
print('RMSE: ', np.sqrt(metrics.mean_absolute_error(y_test,prediction)))


sns.distplot((y_test-prediction),bins=50)

# Conclusion

Coeff_df=pd.DataFrame(lm.coef_,X.columns,columns=['Coeffecient'])
Coeff_df