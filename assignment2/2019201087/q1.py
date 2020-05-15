#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import pickle
import glob
import errno
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler


# In[ ]:


files = glob.glob('/home/ubuntu/Desktop/sem2/smai/assignment2/Datasets/Question-1/cifar-10-batches-py/data_batch_*')


# In[ ]:


print(files)


# In[ ]:





# In[ ]:


final_data = []
final_labels = []
# name = files[0]
# print(name)
i = 0
for name in files:
    try:
        with open(name, 'rb') as f:
            print(name)
            text = pickle.load(f, encoding = 'bytes')
#             print(text)
            labels = text[b'labels']
            data = text[b'data']
            print(data.shape)
            if i == 0:
                 
                final_data = np.array(data)
                final_labels = np.array(labels)
            else:
                final_data = np.vstack((final_data,data))
                final_labels = np.append(final_labels,labels)
                
            print("final_data shape = ",final_data.shape)
            i += 1    
#             print(data.shape)
#             labels = np.array(labels)
#             print("type of labels = ",type(labels)," and type of data = ",type(data))
#             print(labels.shape)
#             print(len(final_data))

    except IOError as exc: #Not sure what error this is
        if exc.errno != errno.EISDIR:
            raise
# final_data = np.asarray(final_data)
print("final data = ",final_data.shape)
print("final_labels = ",final_labels.shape)
print("type of labels = ",type(final_labels)," and type of data = ",type(final_data))
print(final_labels.shape)
pca = PCA(n_components=100)
transformed_data = pca.fit_transform(final_data)
# transformed_data = pca.transform(final_data)
print(transformed_data.shape)
print("PCA was used here ")
train_data = transformed_data[:40000,:]
print(train_data)
train_labels = final_labels[:40000]
test_data = transformed_data[40000:,:]
test_labels = final_labels[40000:]
scaling = MinMaxScaler(feature_range=(-1,1)).fit(train_data)
X_train = scaling.transform(train_data)
X_test = scaling.transform(test_data)
print(X_train)
print('fitting data to classifier')
clf = SVC(kernel = 'linear', C = 2.0,decision_function_shape = 'ovr')
clf.fit(X_train, train_labels.flatten())
print("predicting the test_data")
y_pred = clf.predict(X_test)
print("SVC using PCA",accuracy_score(test_labels,y_pred))
        


            
            
    

