# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 22:02:30 2022

@author: Kush Bhandari
"""

'''
+-----------+----------------+
| Estimator | Model Accuracy |
+-----------+----------------+
|     10    |    86.8852     |
|     12    |    73.7705     |
|     14    |    83.6066     |
|     16    |    86.8852     |
|     18    |    85.2459     |
|     20    |    81.9672     |
|     22    |    85.2459     |
|     24    |    80.3279     |
|     26    |    81.9672     |
|     28    |    81.9672     |
|     30    |    86.8852     |
|     32    |    83.6066     |
|     34    |    83.6066     |
|     36    |    85.2459     |
|     38    |    86.8852     |
|     40    |    83.6066     |
|     42    |    85.2459     |
|     44    |    81.9672     |
|     46    |    85.2459     |
|     48    |    85.2459     |
|     50    |    85.2459     |
|     52    |    86.8852     |
|     54    |    83.6066     |
|     56    |    86.8852     |
|     58    |    83.6066     |
|     60    |    85.2459     |
|     62    |    85.2459     |
|     64    |    85.2459     |
|     66    |    86.8852     |
|     68    |    86.8852     |
|     70    |    85.2459     |
|     72    |    81.9672     |
|     74    |    81.9672     |
|     76    |    86.8852     |
|     78    |    86.8852     |
|     80    |    88.5246     |
|     82    |    85.2459     |
|     84    |    85.2459     |
|     86    |    81.9672     |
|     88    |    83.6066     |
|     90    |    81.9672     |
|     92    |    85.2459     |
|     94    |    86.8852     |
|     96    |    86.8852     |
|     98    |    86.8852     |
+-----------+----------------+ 


The max value is 88.5246
The estimator with max accuracy is 80

'''


import pickle
import pandas as pd 
import numpy as np
from prettytable import PrettyTable
myTable = PrettyTable(["Estimator", "Model Accuracy"])
heart_disease = pd.read_csv('/Users/Kush Bhandari/Desktop/heart.csv') 


#print(heart_disease,'\n\n')

X = heart_disease.drop(['target'] , axis=1)  
Y = heart_disease['target'] 

from sklearn.ensemble import RandomForestClassifier  
clf = RandomForestClassifier(n_estimators=10) 
from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2) 
                                                  
clf.fit(X_train, Y_train) 
y_pred = clf.predict(X_test) 
#print(clf.score(X_train, Y_train),'\n\n') 
#print(clf.score(X_test, Y_test),'\n\n')

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#print(classification_report(Y_test,y_pred),'\n\n')
#print(confusion_matrix(Y_test,y_pred),'\n\n')
#print(accuracy_score(Y_test,y_pred),'\n\n')

np.random.seed(42)

max=0.0
maxi=0;
for i in range(10,100,2):
    print("Trying model with",{i} ,"estimators...")
    clf = RandomForestClassifier(n_estimators=i).fit(X_train,Y_train)
    temp=clf.score(X_test,Y_test)*100
    temp=round(temp,4)
    print("Model accuracy on test set= ",temp)
    if (temp>max):
        max=temp
        maxi=i
    myTable.add_row([i,temp])
         

pickle.dump(clf, open("random_forst_model_1.pkl", "wb"))
loaded_model=pickle.load(open("random_forst_model_1.pkl","rb"))
#print('\n',loaded_model.score(X_test,Y_test))
print('\n\n')

print(myTable,'\n\n')

print('The max value is',max)
print('The estimator with max accuracy is',maxi)