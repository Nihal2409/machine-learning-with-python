
# Random Forest Project 

# Here are what the columns represent:
# * credit.policy: 1 if the customer meets the credit underwriting criteria of LendingClub.com, and 0 otherwise.
# * purpose: The purpose of the loan (takes values "credit_card", "debt_consolidation", "educational", "major_purchase", "small_business", and "all_other").
# * int.rate: The interest rate of the loan, as a proportion (a rate of 11% would be stored as 0.11). Borrowers judged by LendingClub.com to be more risky are assigned higher interest rates.
# * installment: The monthly installments owed by the borrower if the loan is funded.
# * log.annual.inc: The natural log of the self-reported annual income of the borrower.
# * dti: The debt-to-income ratio of the borrower (amount of debt divided by annual income).
# * fico: The FICO credit score of the borrower.
# * days.with.cr.line: The number of days the borrower has had a credit line.
# * revol.bal: The borrower's revolving balance (amount unpaid at the end of the credit card billing cycle).
# * revol.util: The borrower's revolving line utilization rate (the amount of the credit line used relative to total credit available).
# * inq.last.6mths: The borrower's number of inquiries by creditors in the last 6 months.
# * delinq.2yrs: The number of times the borrower had been 30+ days past due on a payment in the past 2 years.
# * pub.rec: The borrower's number of derogatory public records (bankruptcy filings, tax liens, or judgments).

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

loans=pd.read_csv('loan_data.csv')
loans.info()
loans.describe()
loans.head()

# Exploratory Data Analysis

plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist(bins=35,label='Credit policy = 1',alpha=0.6,color='blue')
loans[loans['credit.policy']==0]['fico'].hist(bins=35,label='Credit policy = 0',alpha=0.6,color='red')
plt.legend()

plt.figure(figsize=(10,6))
loans[loans['not.fully.paid']==1]['fico'].hist(bins=35,label='not.fully.paid = 1',alpha=0.6,color='blue')
loans[loans['not.fully.paid']==0]['fico'].hist(bins=35,label='not.fully.paid = 0',alpha=0.6,color='red')
plt.legend()


plt.figure(figsize=(10,6))
sns.countplot(x='purpose',hue='not.fully.paid',data=loans)

sns.jointplot(x='fico',y='int.rate',data=loans)
sns.lmplot(x='fico',y='int.rate',data=loans,hue='credit.policy',col='not.fully.paid')

# Setting up the Data
loans.info()

# Categorical Features

cat_feats=['purpose']
final_data=pd.get_dummies(loans,columns=cat_feats,drop_first=True)
final_data.head()

# Train Test Split
from sklearn.model_selection import train_test_split
X=final_data.drop('not.fully.paid',axis=1)
y=final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Training a Decision Tree Model

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)

# Predictions and Evaluation of Decision Tree

predictions=dtree.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))


# Training the Random Forest model
from sklearn.ensemble import RandomForestClassifier
rfc= RandomForestClassifier(n_estimators=300)
rfc.fit(X_train,y_train)

# Predictions and Evaluation
predict_rf=rfc.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predict_rf))

print(confusion_matrix(y_test,predict_rf))