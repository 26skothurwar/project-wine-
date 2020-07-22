#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
df=pd.read_csv(r'/users/sowmya/downloads/winequality-red.csv', sep=';')


# In[35]:


df.head()


# In[36]:


df.columns


# In[37]:


y=df['quality']
x=df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']]


# In[38]:



df.info()


# In[39]:


#split into train test

from sklearn.model_selection import train_test_split
trainx,testx,trainy,testy=train_test_split(x,y,test_size=0.2,random_state=42)


#applying standard scaling

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
trainx=sc.fit_transform(trainx)
testx=sc.fit_transform(testx)


# In[40]:


df.describe()


# In[41]:


import matplotlib.pyplot as plt
df.hist(bins=10,figsize=(12,10))
plt.show()


# In[42]:


df.plot(kind='density',subplots=True,layout=(4,3),sharex=False)
plt.show()


# In[43]:


#create pivot table
columns=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol','quality']
pivot=df.pivot_table(columns,['quality'],aggfunc='median')
print(pivot)


# In[44]:


#correlation
correlations=df.corr()
print(correlations["quality"].sort_values(ascending=False))


# In[45]:


#colormap

import seaborn as sns
from seaborn import color_palette
colormap=sns.diverging_palette(220,10,as_cmap=True)

# Plot figsize
fig, ax = plt.subplots(figsize=(10, 10))

#generate heatmap,allow annotations and place floats in map
correlations=df.corr()
sns.heatmap(correlations,cmap=colormap,annot=True,fmt=".2f")
ax.set_xticklabels(columns,rotation=45,horizontalalignment='right');
ax.set_yticklabels(columns);
plt.show()


# In[46]:


#Scatterplot matrix

from pandas.plotting import scatter_matrix
sm=scatter_matrix(df,figsize=(12,12),diagonal='hist')

#change label rotation
[s.xaxis.label.set_rotation(45)for s in sm.reshape(-1)]
[s.yaxis.label.set_rotation(0)for s in sm.reshape(-1)]

#may need to offset label when rotating to prevent overlap of figure

[s.get_yaxis().set_label_coords(-.6,0.5)for s in sm.reshape(-1)]

#hide all ticks
[s.set_xticks(()) for s in sm.reshape(-1)]
[s.set_yticks(()) for s in sm.reshape(-1)]
plt.show()


# In[47]:


df.quality.nunique()


# In[ ]:


# Dividing wine as good and bad by giving the limit for the quality
from sklearn.preprocessing import LabelEncoder

df['quality']=pd.cut(x=df['quality'], bins = (2,6,8), labels = ["bad","good"])
# Now lets assign a labels to our quality variable
label_quality = LabelEncoder()
# Bad becomes 0 and good becomes 1
df['quality'] = label_quality.fit_transform(df['quality'])
print(df['quality'].value_counts())
sns.countplot(df['quality'])
plt.show()


# In[49]:


#prepare configuration for cross validation test harness
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier,LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn import model_selection



seed=7

#prepare models
models=[]
models.append(('SupportVectorClassifer',SVC()))
models.append(('stochasticGradientDecentC',SGDClassifier()))
models.append(('RandomForestClassifier',RandomForestClassifier()))
models.append(('DecisionTreeClassifier',DecisionTreeClassifier()))
models.append(('GaussianNB',GaussianNB()))
models.append(('KNeighborsClassifiers',KNeighborsClassifier()))
models.append(('LogisticRegression',LogisticRegression()))

#evaluate each model in turn
results=[]
names=[]
scoring='accuracy'
for name,model in models:
    kfold=model_selection.KFold(n_splits=10,random_state=seed)
    cv_results=model_selection.cross_val_score(model,trainx,trainy,cv=kfold,scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg="%s:%f(%f)"%(name,cv_results.mean(),cv_results.std())
    print(msg)
    
#boxplot algorithm comparision
fig=plt.figure(figsize=(12,10))
fig.suptitle('Algorithm Comparison')
ax=fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()





# In[ ]:





# In[ ]:




