#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import os
import re
import pandas as pd
import codecs
import numpy as np
import sys
from sklearn.svm import SVC


# In[ ]:





# In[ ]:


batch_size = 128
num_classes = 10
epochs = 20


# In[ ]:





# In[ ]:


class CNN:
    
    def reshape_data2d(self, X_train):
        data = X_train.reshape(X_train.shape[0], 28, 28, 1)
#         input_shape = (28,28, 1)
        data = data.astype('float32')
        data /= 255
        #print('x_train shape:', data.shape)
        #print(data.shape[0], 'train samples')
        return data
        
        
    def train(self, X_train, Y_train):
        
        
        x_train = self.reshape_data2d(X_train)
        #print(x_train.shape)
        y_train = keras.utils.to_categorical(Y_train, num_classes)
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=(28,28, 1)))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(num_classes, activation='softmax'))
        
        self.model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
        
        self.model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs)
        
        
    def predict(self, X_test):
        labels = []
        x_test = self.reshape_data2d(X_test)
        preds = self.model.predict(x_test)
        for i in range(len(preds)):
            labels.append(np.argmax(preds[i]))
        return labels

        


# In[ ]:


class MLP:
    
    def reshape1d(self, X_train):
        data = X_train.reshape(X_train.shape[0], 784)
        data = data.astype('float32')
        data /= 255
        return data
    
    def train(self, X_train, Y_train):
        
        x_train = self.reshape1d(X_train)
        y_train = keras.utils.to_categorical(Y_train, num_classes)
        
        self.model1 = Sequential()

        self.model1.add(Dense(512, activation="relu",input_shape=(784,)))
        self.model1.add(Dense(512,activation="relu"))
        self.model1.add(Dense(num_classes, activation='softmax'))
        self.model1.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
        self.model1.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs)
        
    def predict(self, X_test):
        labels = []
        x_test = self.reshape1d(X_test)
        preds = self.model1.predict(x_test)
        for i in range(len(preds)):
            labels.append(np.argmax(preds[i]))
        return labels


# In[ ]:





# In[ ]:





# In[ ]:


# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train.shape


# In[ ]:


# class SVM_algo:
    
#     def reshape1d(self, X_train):
#         data = X_train.reshape(X_train.shape[0], 784)
#         data = data.astype('float32')
#         data /= 255
#         return data
    
#     def train(self, X_train, Y_train):
        
#         x_train = self.reshape1d(X_train)
#         self.clf = SVC(gamma='auto')
#         self.clf.fit(x_train,Y_train)
        
#     def predict_values(self, X_test):
#         x_test = X_test.reshape(X_test.shape[0], 784)
#         preds = self.clf.predict(x_test)
#         return preds


# In[ ]:


def parse_file(path):
    dataset = pd.DataFrame([])
    
    with open (path, 'rb') as file:
        #print('open file')
        file_data = file.read()
#         print(file_data)
        '''Number of images'''
        length = int(codecs.encode(file_data[4:8], 'hex'), 16)
        #print(length)
        ''' number of rows (dimension 1)'''
        rows = int(codecs.encode(file_data[8:12], 'hex'), 16)
        #print(rows)
        ''' number of columns (dimension 2)'''
        columns = int(codecs.encode(file_data[12:16], 'hex'), 16)
        #print(columns)
        data = np.frombuffer(file_data, dtype = np.uint8, offset = 16)
        data = data.reshape(length, rows, columns)
    
    return data


def parse_file_for_labels(path):
    dataset = pd.DataFrame([])
    with open(path, 'rb') as file:
        file_data = file.read()
        length = int(codecs.encode(file_data[4:8], 'hex'), 16)
        data = np.frombuffer(file_data, dtype = np.uint8, offset = 8)
        data = data.reshape(length)
    
    return data
        


# In[ ]:



def read_files(filepath):
    

    files_present = os.listdir(filepath)
    for file in files_present:
        if(re.search('train-images', file) != None):
            train_images= filepath + file
            #print(train_images)
            dataset_train = parse_file(train_images)
            #print(dataset_train.shape)

        if(re.search('train-labels', file) != None):
            train_labels = filepath + file
            #print(train_labels)
            labelset_train = parse_file_for_labels(train_labels)
            #print(labelset_train.shape)

        if(re.search('t10k-images', file) != None):
            test_images = filepath + file
            #print(test_images)
            dataset_test = parse_file(test_images)
            #print(dataset_test.shape)

        if(re.search('t10k-labels', file) !=  None):
            test_labels = filepath + file
            #print(test_labels)
            labelset_test = parse_file_for_labels(test_labels)
            #print(labelset_test.shape)
            
    return dataset_train, dataset_test, labelset_train, labelset_test
        


# In[ ]:


X_train, X_test, Y_train, Y_test = read_files(sys.argv[1])


# In[ ]:


cnn = CNN()
cnn.train(X_train, Y_train)
labels = cnn.predict(X_test)


from sklearn.metrics import accuracy_score,confusion_matrix
print(accuracy_score(Y_test, labels))
print(confusion_matrix(Y_test, labels))


# In[ ]:


mlp = MLP()
mlp.train(X_train, Y_train)
labels = mlp.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix
print(accuracy_score(Y_test, labels))
print(confusion_matrix(Y_test, labels))


# In[ ]:


'''svm = SVM_algo()
svm.train(X_train,Y_train)
labels = svm.predict_values(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix
print(accuracy_score(Y_test, labels))
print(confusion_matrix(Y_test, labels))'''


# In[ ]:


x_train = X_train.reshape(X_train.shape[0], 784)
x_test = X_test.reshape(X_test.shape[0], 784)


# In[ ]:


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
#print('x_train shape:', x_train.shape)
# print('y_train shape:', y_train.shape)
#print(x_train.shape[0], 'train samples')
#print(x_test.shape[0], 'test samples')


# In[ ]:


clf = SVC(gamma='auto')
clf.fit(x_train,Y_train)


# In[ ]:


preds = clf.predict(x_test)

from sklearn.metrics import accuracy_score,confusion_matrix
print(accuracy_score(Y_test, preds))
print(confusion_matrix(Y_test, preds))



