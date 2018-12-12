# Naive Bayes

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('DataSetMod.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,len(dataset.columns)-1]
from sklearn import preprocessing
labelencoder = preprocessing.LabelEncoder()
y = labelencoder.fit_transform(y)
#y = dataset.iloc[:,len(dataset.columns)-1]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/4, random_state = 0)

from sklearn.decomposition import PCA
pca = PCA(n_components = 1600)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score

acc_score = accuracy_score(y_test, y_pred)

