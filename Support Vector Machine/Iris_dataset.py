
# Support Vector Machines Project For IRIS Dataset

# The Iris Setosa
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
Image(url,width=300, height=300)

# The Iris Versicolor
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg'
Image(url,width=300, height=300)

# The Iris Virginica
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
Image(url,width=300, height=300)

# The iris dataset contains measurements for 150 iris flowers from three different species.
# The three classes in the Iris dataset:
#     Iris-setosa (n=50)
#     Iris-versicolor (n=50)
#     Iris-virginica (n=50)
# The four features of the Iris dataset:
#     sepal length in cm
#     sepal width in cm
#     petal length in cm
#     petal width in cm
# Get the data
import seaborn as sns
iris=sns.load_dataset('iris')
iris.head()

# Exploratory Data Analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.pairplot(iris,hue='species')
setosa=iris[iris['species']=='setosa']
sns.kdeplot(setosa['sepal_width'],setosa['sepal_length'],shade=True,cmap='plasma',shade_lowest=False)

# Train Test Split
from sklearn.model_selection import train_test_split
X=iris.drop('species',axis=1)
y=iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train a Model
from sklearn.svm import SVC
model=SVC()
model.fit(X_train,y_train)

# Model Evaluation
predictions=model.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

# Gridsearch
from sklearn.grid_search import GridSearchCV
param_grid ={'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(X_train,y_train)

grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))

print(classification_report(y_test,grid_predictions))