
# K Nearest Neighbors Project 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Get the Data

df=pd.read_csv('KNN_Project_Data')
df.head()

# EDA
sns.pairplot(df,hue='TARGET CLASS')

# # Standardize the Variables

from sklearn.preprocessing import StandardScaler
# Create a StandardScaler() object called scaler.**
ss=StandardScaler()

ss.fit(df.drop('TARGET CLASS',axis=1))

ss_features=ss.transform(df.drop('TARGET CLASS',axis=1))

df_feat=pd.DataFrame(ss_features,columns=df.columns[:-1])
df_feat.head()

# Train Test Split

from sklearn.model_selection import train_test_split
X=df_feat
y=df['TARGET CLASS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Using KNN
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)

# Fit this KNN model to the training data.

knn.fit(X_train,y_train)
# Predictions and Evaluations

predictions=knn.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(predictions,y_test))
print(classification_report(y_test,predictions))

# Choosing a K Value

error_rate=[]
for i in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_k=knn.predict(X_test)
    error_rate.append(np.mean(pred_k != y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue',linestyle='dashed',marker='o',
        markerfacecolor='red',markersize=10)
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.title('K vs Error')

# Retrain with new K Value

knn=KNeighborsClassifier(n_neighbors=30)
knn.fit(X_train,y_train)
predictions=knn.predict(X_test)
print(confusion_matrix(predictions,y_test))
print(classification_report(y_test,predictions))