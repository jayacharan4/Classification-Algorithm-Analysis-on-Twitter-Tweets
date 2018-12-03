# Support Vector Machine (SVM)

# Importing the libraries
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
"""
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
"""

from sklearn.decomposition import PCA
pca = PCA(n_components = 2500)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
#explained_variance = pca.explained_variance_ratio_

"""
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 500)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)
"""
#for knn
"""
from sklearn.neighbors import KNeighborsClassifier as KNN
classifier = KNN(n_neighbors=4)
classifier.fit(X_train, y_train)
"""
# Fitting SVM to the Training set

from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test, y_pred)