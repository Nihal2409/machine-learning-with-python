
# K Means Clustering Project 
# We will use a data frame with 777 observations on the following 18 variables.
# * Private A factor with levels No and Yes indicating private or public university
# * Apps Number of applications received
# * Accept Number of applications accepted
# * Enroll Number of new students enrolled
# * Top10perc Pct. new students from top 10% of H.S. class
# * Top25perc Pct. new students from top 25% of H.S. class
# * F.Undergrad Number of fulltime undergraduates
# * P.Undergrad Number of parttime undergraduates
# * Outstate Out-of-state tuition
# * Room.Board Room and board costs
# * Books Estimated book costs
# * Personal Estimated personal spending
# * PhD Pct. of faculty with Ph.D.’s
# * Terminal Pct. of faculty with terminal degree
# * S.F.Ratio Student/faculty ratio
# * perc.alumni Pct. alumni who donate
# * Expend Instructional expenditure per student
# * Grad.Rate Graduation rate

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

df=pd.read_csv('College_Data',index_col=0)
df.head()
df.info()
df.describe()

# EDA

sns.scatterplot(x='Room.Board',y='Grad.Rate',data=df,hue='Private')

sns.scatterplot(x='Outstate',y='F.Undergrad',data=df,hue='Private')

plt.figure(figsize=(14,6))
df[df['Private']=='Yes']['Outstate'].plot.hist(bins=30,alpha=0.4)
df[df['Private']=='No']['Outstate'].plot.hist(bins=30,alpha=0.4)


plt.figure(figsize=(14,6))
df[df['Private']=='Yes']['Grad.Rate'].plot.hist(bins=30,alpha=0.4)
df[df['Private']=='No']['Grad.Rate'].plot.hist(bins=30,alpha=0.4)


df[df['Grad.Rate']>100]
df['Grad.Rate']['Cazenovia College']=100

plt.figure(figsize=(14,6))
df[df['Private']=='Yes']['Grad.Rate'].plot.hist(bins=30,alpha=0.4)
df[df['Private']=='No']['Grad.Rate'].plot.hist(bins=30,alpha=0.4)

# K Means Cluster Creation

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

kmeans =KMeans(n_clusters=2)

X=df.drop('Private',axis=1)
y=df['Private']
kmeans.fit(X)
kmeans.cluster_centers_

# Evaluation

def converter(cluster):
    if cluster =='Yes':
        return 1
    else:
        return 0
df['Cluster']=df['Private'].apply(converter)
df.head()
from sklearn.metrics import confusion_matrix,classification_report

print(confusion_matrix(df['Cluster'],kmeans.labels_))
print('\n')
print(classification_report(df['Cluster'],kmeans.labels_))