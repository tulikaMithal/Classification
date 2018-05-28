# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 23:42:56 2018

@author: lenovo
"""
import pandas as pd
import sys
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold


def plot_confusion_matrix(cm, classes, ImageFileName,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(ImageFileName)
    plt.close()
    
    
def NeuralNetwork(train_x_data, train_y_data, test_x_data, test_y_data):
    mlp = MLPClassifier(hidden_layer_sizes=(30,30,30), max_iter=200)
    mlp.fit(train_x_data,train_y_data)
    predictions = mlp.predict(test_x_data)
    test_y_data = [item.replace(".","") for item in test_y_data]
    print("\nAccuracy Score obtained using Neural Network on test data: ")
    print(accuracy_score(test_y_data,predictions))
    cnf = confusion_matrix(test_y_data,predictions)
    plot_confusion_matrix(cnf,['<=50k','>50k'],"Neural_Network_Conf_Matrix.png")
    
    
    
def supportVector(train_x_data, train_y_data, test_x_data, test_y_data):
    svc_radial = svm.SVC()
    svc_radial.fit(train_x_data, train_y_data)
    predicted= svc_radial.predict(test_x_data)
    test_y_data = [item.replace(".","") for item in test_y_data]
    print("\nAccuracy Score obtained using Support Vector Machine on test data : ")
    print(accuracy_score(test_y_data, predicted))
    cnf = confusion_matrix(test_y_data,predicted)
    plot_confusion_matrix(cnf,['<=50k','>50k'], "Support_Vector_Conf_Matrix.png")
    
def knn(train_x_data, train_y_data, test_x_data, test_y_data):    
  
    KNN = KNeighborsClassifier(n_neighbors=13)
    KNN.fit(train_x_data,train_y_data)
    predicted = KNN.predict(test_x_data)
    test_y_data = [item.replace(".","") for item in test_y_data]
    print("\nAccuracy Score obtained using KNN on test data: ")
    print(accuracy_score(test_y_data, predicted))
    cnf = confusion_matrix(test_y_data,predicted)
    plot_confusion_matrix(cnf,['<=50k','>50k'],"KNN_Conf_Matrix.png")
    
# Reading training data
train_data = pd.read_csv(sys.argv[1], header = None)

# Reading test data
test_data = pd.read_csv(sys.argv[2], header = None)

# Replacing '?' with NaN 
train_data = train_data.replace(' ?', np.NaN)
# Dropping rows with NaN
train_data = train_data.dropna()

train_x_data = train_data.iloc[:,0:-1]
train_y_data = train_data.iloc[:,-1]

test_x_data = test_data.iloc[:,0:-1]
test_y_data = test_data.iloc[:,-1]


# Converting data to numerical values 
le = preprocessing.LabelEncoder()
train_x_data = train_x_data.apply(le.fit_transform)
test_x_data = test_x_data.apply(le.fit_transform)


scaler = StandardScaler()

scaler.fit(train_x_data)
scaler.fit(test_x_data)

train_x_data = scaler.transform(train_x_data)
test_x_data = scaler.transform(test_x_data)

NeuralNetwork(train_x_data, train_y_data, test_x_data, test_y_data)
supportVector(train_x_data, train_y_data, test_x_data, test_y_data)
knn(train_x_data, train_y_data, test_x_data, test_y_data)
    
        
#import seaborn as sns
# plot data
#colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral','red','green','blue','orange','white','brown']
#train_data[1].value_counts().plot(kind='pie',title='Workclass',colors=colors)
#plt.show()
#
#print(train_data[0].value_counts())
#train_data[0].plot(kind='hist', bins=8, title='Age', facecolor='blue', alpha=0.5, normed=1)
#plt.show()
#
## plot data
#colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral','red','green','blue','orange','white','brown']
#train_data[9].value_counts().plot(kind='pie',title='Workclass',colors=colors)
#plt.show()

#sns.FacetGrid(train_data, hue=14, size=5) \
#   .map(plt.scatter, 0, 1) \
#   .add_legend()
#plt.show()


