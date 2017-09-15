 
import os
os.chdir('C:/Users/aksha/Desktop/Machine learning/titanic_project')

import numpy as np
import pandas as pd


#importing the dataset
dataset = pd.read_csv('titanic_train.csv')

#preview of the data
dataset_train.head()

#overall info of the dataset
dataset_train.info()

#setting the training data
X_train = dataset.iloc[:800,2:12].values
y_train = dataset.iloc[:800,1:2].values

#setting the testing data
X_test = dataset.iloc[800:,2:12].values
y_test = dataset.iloc[800:,1:2].values
                           
#Missing data analysis
ind_null_train = np.zeros((X_train.shape[1],1))
percentage_missing_train = np.zeros((X_train.shape[1],1))

for i in range(0,X_train.shape[1]):
    ind_null_train[i] = len(np.where(pd.isnull(X_train[:,i]))[0])
    percentage_missing_train[i] = ind_null_train[i]/len(X_train[:,i])*100


ind_null_test = np.zeros((X_test.shape[1],1))
percentage_missing_test = np.zeros((X_test.shape[1],1))

for i in range(0,X_test.shape[1]):
    ind_null_test[i] = len(np.where(pd.isnull(X_test[:,i]))[0])
    percentage_missing_test[i] = ind_null_test[i]/len(X_test[:,i])*100


#deleting the name,ticket and Cabin columns as it does not provide any useful information
X_train = np.delete(X_train, [1,6,8], axis=1)
X_test = np.delete(X_test, [1,6,8], axis=1)


#filling the missing data with most frequent value
count_Q_train = np.count_nonzero(X_train[:,6] == 'Q')
count_S_train = np.count_nonzero(X_train[:,6] == 'S')
count_C_train = np.count_nonzero(X_train[:,6] == 'C')
freq = max(count_Q_train, count_S_train, count_C_train)
if freq == count_Q_train:
    X_train[np.where(pd.isnull(X_train[:,6])),6] = 'Q'
elif freq == count_S_train:
    X_train[np.where(pd.isnull(X_train[:,6])),6] = 'S'
elif freq == count_C_train:
    X_train[np.where(pd.isnull(X_train[:,6])),6] = 'C'


count_Q_test = np.count_nonzero(X_test[:,6] == 'Q')
count_S_test = np.count_nonzero(X_test[:,6] == 'S')
count_C_test = np.count_nonzero(X_test[:,6] == 'C')
freq = max(count_Q_test, count_S_test, count_C_test)
if freq == count_Q_test:
    X_test[np.where(pd.isnull(X_test[:,6])),6] = 'Q'
elif freq == count_S_test:
    X_test[np.where(pd.isnull(X_test[:,6])),6] = 'S'
elif freq == count_C_test:
    X_test[np.where(pd.isnull(X_test[:,6])),6] = 'C'    
            

#filling the missing data with mean of that feature
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy = "mean",axis=0)
imputer.fit(X_train[:,2:3])
X_train[:,2:3] = imputer.transform(X_train[:,2:3])
X_test[:,2:3] = imputer.transform(X_test[:,2:3])


#label encoding
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder = LabelEncoder()
X_train[:,1] = labelencoder.fit_transform(X_train[:,1])
X_test[:,1] = labelencoder.transform(X_test[:,1])
labelencoder = LabelEncoder()
X_train[:,6] = labelencoder.fit_transform(X_train[:,6])
X_test[:,6] = labelencoder.transform(X_test[:,6])

onehotencoder = OneHotEncoder(categorical_features = [6])
X_train = onehotencoder.fit_transform(X_train).toarray()
X_test = onehotencoder.transform(X_test).toarray()


#fitting logistic regression model
from sklearn.linear_model import LogisticRegression
classifier_logreg = LogisticRegression(random_state = 0)
classifier_logreg.fit(X_train,y_train)

#predicting the results for logistic regression model
y_pred_logreg = classifier_logreg.predict(X_test)

#Creating confusion matrix for logistic regression
from sklearn.metrics import confusion_matrix
cm_logreg = confusion_matrix(y_test, y_pred_logreg)

#fitting KNN classifier
from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier(n_neighbors = 7, metric = 'minkowski', p=2)
classifier_knn.fit(X_train,y_train)

#predicting the results for KNN classifier
y_pred_knn = classifier_knn.predict(X_test)

#creating confusion matrix for knn
cm_knn = confusion_matrix(y_test,y_pred_knn)

#fitting SVM classifier
from sklearn.svm import SVC
classifier_svm = SVC(kernel='linear', random_state=0)
classifier_svm.fit(X_train,y_train)

#predicting the results
y_pred_svm = classifier_svm.predict(X_test)

#creating confusion matrix for SVM
cm_svm = confusion_matrix(y_test,y_pred_svm)

#fitting kernel SVM
classifier_ksvm = SVC(kernel='rbf', random_state=0)
classifier_ksvm.fit(X_train,y_train)

#predicting the results
y_pred_ksvm = classifier_ksvm.predict(X_test) 

#creating confusion matrix for ksvm
cm_ksvm = confusion_matrix(y_test,y_pred_ksvm)

#fitting naive Bayes theorem
from sklearn.naive_bayes import GaussianNB
classifier_nb = GaussianNB()
classifier_nb.fit(X_train,y_train)

#predicting the results for naive bayes 
y_pred_nb = classifier_nb.predict(X_test)

#creating the confusion matrix for naive bayes
cm_nb = confusion_matrix(y_test,y_pred_nb)

#Fitting decision tree classifier
from sklearn.tree import DecisionTreeClassifier
classifier_dt = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier_dt.fit(X_train,y_train)

#predicting the results
y_pred_dt = classifier_dt.predict(X_test)

#creating confusion matrix for decsion tree
cm_dt = confusion_matrix(y_test, y_pred_dt)

#fitting random forest
from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(n_estimators=10, criterion = 'entropy', random_state=0)
classifier_rf.fit(X_train,y_train)

#predicting the results
y_pred_rf = classifier_rf.predict(X_test)

#confusion matrix for random forest
cm_rf = confusion_matrix(y_test,y_pred_rf)



