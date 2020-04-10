#!/usr/bin/env python
# coding: utf-8

# In[253]:


import numpy as np 
import pandas as pd 
import math 
import matplotlib.pyplot as plt 
from collections import Counter 
from sklearn.metrics import accuracy_score


# In[ ]:





# In[251]:


class KNNClassifier():
    
    def __init__(self):
#        print("")
        global k
        k = 3
        label_training = np.empty(2)
#         training_data = np.empty(2)
        X_train = np.empty(2)
#         validation_data = np.empty(2)
#         label_validation = np.empty(1)

    def predict_values_for_validation(self, train_data, test_data, label_data, k, distance_function):
        final_labels = []
#        print(test_data.shape)
#        print(train_data.shape)
#         print(labels_valid.shape)
        for j in range(len(test_data)):
#            print(j)
            distance_labels_and_indexes = []
            labels = []
            dist = 0
            for i in range(len(train_data)):
#                 print(train_data.iloc[i,1:])
#                 print(test_data.iloc[j,:])
#                 print("train_data : ", train_data[i,:].shape)
#                 print("test_data : ", test_data[j,:].shape)
                dist = self.euclidean(train_data[i,:], test_data[j,:])
            # #         print(test[0][3])
#                 print(dist)
                distance_labels_and_indexes.append((dist, i))
            sorted_dist = sorted(distance_labels_and_indexes)
            first_k_values = sorted_dist[:k]
            # print(len(first_k_values))
            for dist,i in first_k_values:
                index = i
                labels.append(label_data[i])
              # print("index", index, "label",data[i][0])
            label =  self.mode(labels)
#            print("i ", j, " label ",label," org ")
            final_labels.append(label)
        return final_labels


    def euclidean(self,a, b):
        dist = np.linalg.norm(a-b)
        return dist 

    def manhattan(self,ippt,testpt):
        sum_ = 0
        for i in range(len(testpt)):
            sum_ += abs(ippt[i] - testpt[i])

        return sum_
    
    def mode(self,labels):
        return Counter(labels).most_common(1)[0][0]




    def handle_missing_values(self,data):
#         df = np.array(data)
        for i in range(data.shape[1]):
            item_counts = Counter(data[:,i])
#             print(item_counts)
            max_item = item_counts.most_common()[0][0]
#             print("max_item = ",max_item)
#             items_counts = data[:,i].value_counts()
#             max_item = items_counts.max()
            # print(data.iloc[:,11].value_counts())

        #replace with mode
        for i in range(0,data.shape[0]):
            if data[i,10] == '?':
                  data[i,10] = 'b'
                    
        return data
    # print(data.iloc[:,10].value_counts())'''
    
    
    def predict(self,test_data_path):
        test_data = pd.read_csv(test_data_path, header=None)
        test_data = test_data.values
        temp_test_data = test_data
        temp_test_data = self.handle_missing_values(temp_test_data)
        temp_test_data = self.fit_transform(temp_test_data,lists)
#        print(temp_test_data,' ', temp_test_data.shape)
#         self.test_data = self.test_data[:10]
#         self.test_data = self.test_data[]
#        print("shape = ",self.X_train.shape)
#        print("shape = ",temp_test_data.shape)
#         self.test_data = test_data.to_numpy()
#         self.train_data1 = self.train_data1.to_numpy()
        labels = self.predict_values_for_validation(self.X_train, temp_test_data, self.label_training, k, distance_function='euclidean')
        return labels
        
    def unique_keys(self,data):
        df = np.array(data)
        # print(df.shape[1])
        dictionary = []
        lists = []
        for i in range(0,df.shape[1]):
            dictval = {}
            set1= set()
            for j in range(len(df)):
                set1.add(df[j][i])
            count=0
            li = []
            if i == 6:
                set1.add('n')
                set1.add('d')
            if i == 7:
                set1.add('d')
            if i == 11:
                set1.add('u')
                set1.add('z')
            if i == 16:
                set1.add('u')
            if i == 19:
                set1.add('c')
                set1.add('s')
                set1.add('z')
            val = 0
            for j in set1:
                dictval[j] = val
                li.append((j,val))
                val += 1
                count +=1
            lists.append(dictval)
            dictionary.append(li)
        return lists
    
    def fit_transform(self, data, lists):
#        print("shape of data ",data.shape)
        one_hot = np.empty((data.shape[0],0),dtype='int')
#         print(one_hot)
        nrows = data.shape[0]
#        print("len of lists ", len(lists))
        for i in range(len(lists)):
            ncols = len(lists[i])
            df = np.zeros((nrows, ncols),dtype='int')
            j = i
            tempdict = lists[i]
#             print(tempdict)
            for k in range(nrows):
#                 print(k)
#                 print(data.iloc[k,j])
                index = tempdict.get(data[k,j])
#                 print(index)
                df[k,index] = 1
            one_hot = np.concatenate((one_hot,df),axis=1)
#         print("one-hot shape ",one_hot.shape)
        df = np.zeros((nrows,1),dtype='int')
        tempdict = lists[i]
        for k in range(nrows):
            index = tempdict.get(data[k,j])
            df[k,0] = int(index)
        one_hot = np.concatenate((df,one_hot),axis=1)
        # print(one_hot.shape)
        return one_hot
        
    
    def train(self, data_path):
        data = pd.read_csv(data_path)
        data = data.values
#        print(data.shape)
        
        self.label_training = data[:,0]
        training_data = data[:,1:]
        self.X_train = self.handle_missing_values(training_data)
        global lists 
        lists = self.unique_keys(data)
#         print(lists)
        label_lists = lists[0]
        lists = lists[1:]
#        print(lists)
        self.X_train = self.fit_transform(training_data,lists)
#        print(self.X_train,' ', self.X_train.shape)
        '''training_data = data[:4490, :]
        print(training_data.shape)
        validation_data = data[4490:,]
        print(validation_data.shape)
        label_validation = validation_data[:,0]
        self.label_training = training_data[:,0]
        print(label_validation.shape)
        print(self.label_training.shape)
        training_data = training_data[:, 1:]
        print(training_data.shape)
        validation_data = validation_data[:,1:]
        print(validation_data.shape)
        
        
        #handling missing data
        training_data = self.handle_missing_values(training_data)
        print(Counter(training_data[:,10]))
#         print(training_data[:,10].value_counts())
        validation_data = self.handle_missing_values(validation_data)
        print(Counter(validation_data[:,10]))
        lists = self.unique_keys(data)
#         print(lists)
        label_lists = lists[0]
        lists = lists[1:]
        print(lists)
        
        
        self.X_train = self.fit_transform(training_data,lists)
        print(self.X_train,' ', self.X_train.shape)
        X_validation = self.fit_transform(validation_data,lists)
        print(X_validation,' ', X_validation.shape)
#         label_lists = lists[0]
#         lists = lists[1:]
#         print(lists)
        accuracy = self.predict_values_for_validation(self.X_train, X_validation,self.label_training, k  , distance_function = 'euclidean') 
        print (accuracy_score(label_validation, accuracy))'''
    
    


# In[ ]:





# In[252]:



'''knn = KNNClassifier()
knn.train("/home/ubuntu/Desktop/smai/Datasets/q2/train.csv")
pred = knn.predict("/home/ubuntu/Desktop/smai/Datasets/q2/test.csv")
#print(pred)
test_labels = list()
with open("/home/ubuntu/Desktop/smai/Datasets/q2/test_labels.csv") as f:
    for line in f:
        test_labels.append(line.strip())
print (accuracy_score(test_labels, pred))'''
    

