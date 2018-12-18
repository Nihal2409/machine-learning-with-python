
# Natural Language Processing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

yelp = pd.read_csv('yelp.csv')
yelp.head()
yelp.info()
yelp.describe()

yelp['text length']= yelp['text'].apply(len)
# EDA

g = sns.FacetGrid(yelp,col='stars')
g.map(plt.hist,'text length')
sns.boxplot(x='stars',y='text length',data=yelp)
sns.countplot(x='stars',data=yelp)
yelp.groupby('stars').mean()

# Using the corr() method on that groupby dataframe to produce this dataframe:

yelp.groupby('stars').mean().corr()
sns.heatmap(data=yelp.groupby('stars').mean().corr(),annot=True)

# NLP Classification Task
def classi(stars):
    if stars ==1 or stars ==5:
        return True
    else:
        return False
		
yelp_class=yelp[yelp['stars'].apply(classi)]
X=yelp_class['text']
y = yelp_class['stars']

# Import CountVectorizer and create a CountVectorizer object.
from sklearn.feature_extraction.text import CountVectorizer
CV = CountVectorizer()

CV.fit(X)
X = CV.transform(X)

# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Training a Model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train,y_train)

# Predictions and Evaluations

prediction = nb.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,prediction))
print(confusion_matrix(y_test,prediction))
print('\n')
print(classification_report(y_test,prediction))

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('CV',CountVectorizer()),
    ('tfidf',TfidfTransformer()),
    ('classifier',MultinomialNB())
])
# Using the Pipeline

X=yelp_class['text']
y = yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

pipeline.fit(X_train,y_train)

# Predictions and Evaluation
predict_tf = pipeline.predict(X_test)
print(confusion_matrix(y_test,predict_tf))
print('\n')
print(classification_report(y_test,predict_tf))
