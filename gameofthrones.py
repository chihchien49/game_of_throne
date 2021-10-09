#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn import datasets
battles = pd.read_csv("battles.csv")
deaths = pd.read_csv("character-deaths.csv")
predictions = pd.read_csv("character-predictions.csv")
deaths["Death Year"] = deaths["Death Year"].fillna(0)
deaths.loc[deaths["Death Year"] != 0, "Death Year"] = 1

got=deaths.iloc[:,[8]]
dwd=deaths.iloc[:,[12]]

deaths.head(5)


# In[2]:


alle=pd.get_dummies(deaths["Allegiances"])
alle.head(5)
df = pd.concat( [alle,got,dwd], axis=1 )
df.head(5)


# In[3]:


y=deaths.iloc[:,[2]]
X_train, X_test, y_train,y_test = train_test_split(df, y, test_size=0.25,random_state = 42)


# In[4]:


clf = tree.DecisionTreeClassifier(criterion="entropy",max_depth = 20)
clf = clf.fit(X_train, y_train)
y_test_predicted = clf.predict(X_test)


# In[5]:


from sklearn.metrics import accuracy_score,precision_score, recall_score
y_pred = y_test_predicted
y_true = y_test
accuracy_score(y_true, y_pred)



# In[10]:


from sklearn.metrics import confusion_matrix
y_true = y_test
y_pred = y_test_predicted
confusion_matrix(y_true, y_pred,labels=[1, 0])


# In[16]:


precision_score(y_true, y_pred)


# In[17]:


recall_score(y_true, y_pred)


# In[18]:


import pydotplus
dot_data = tree.export_graphviz(clf, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("tree.pdf")


# In[ ]:




