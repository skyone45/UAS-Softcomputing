
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


import keras
from keras.models import Sequential
from keras.layers import Dense


classifier = Sequential()


classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:-1].values

Y = dataset.iloc[:, 13].values

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
labelencoder_country_X = LabelEncoder()
labelencoder_gender_X = LabelEncoder()

X[:, 1] = labelencoder_country_X.fit_transform(X[:, 1])
X[:, 2] = labelencoder_gender_X.fit_transform(X[:, 2])

ohe = OneHotEncoder(categorical_features = [1])
X = ohe.fit_transform(X).toarray()

X = X[:, 1:]

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
Xtrain = sc.fit_transform(Xtrain)
Xtest = sc.transform(Xtest)

classifier.fit(Xtrain, ytrain, batch_size = 10, nb_epoch = 100)

ypred = classifier.predict(Xtest)
ypred = (ypred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest, ypred)