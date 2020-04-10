#!/usr/bin/env python
# coding: utf-8

# In[174]:


import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


# In[ ]:





# In[172]:


class KNNClassifier():
    train_data = np.empty(2)
    validation_data = np.empty(2)
    test_data = np.empty(2)
    train_data1 = np.empty(2)
    k = 3
    def _init__():
        print("")
        
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
    
    
#     def predict_values_for_testing(self, train_data, test_data, k, distance_function = 'euclidean'):
#         final_labels = []
#         print(test_data.shape)
#         print(train_data.shape)
#         for j in range(test_data.shape[0]):
#             print(j)
#             distance_labels_and_indexes = []
#             labels = []
#             dist = 0
#             for i in range(train_data.shape[0]):
# #                 print(i)
#                 dist = self.euclidean(train_data.iloc[i,1:], test_data.iloc[j,:])
#                 print(dist)
        
#         return final_labels
    
    
    def predict(self,test_data_path):
        self.test_data = pd.read_csv(test_data_path, header=None)
#         self.test_data = self.test_data[:10]
#         self.test_data = self.test_data[]
#        print("shape = ",self.train_data1.shape)
#        print("shape = ",self.test_data.shape)
        self.test_data = self.test_data.to_numpy()
        self.train_data1 = self.train_data1.to_numpy()
        labels = self.predict_values_for_validation(self.train_data1, self.test_data, self.k, distance_function='manhattan')
        return labels
        
    def predict_values_for_validation(self, train_data, test_data, k, distance_function):
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
                dist = self.euclidean(train_data[i,1:], test_data[j,:])
            # #         print(test[0][3])
#                 print(dist)
                distance_labels_and_indexes.append((dist, i))
            sorted_dist = sorted(distance_labels_and_indexes)
            first_k_values = sorted_dist[:k]
            # print(len(first_k_values))
            for dist,i in first_k_values:
                index = i
                labels.append(train_data[i][0])
              # print("index", index, "label",data[i][0])
            label =  self.mode(labels)
 #           print("i ", j, " label ",label," org ")
            final_labels.append(label)
        return final_labels
        
        
        
    def train(self,train_path):
        self.train_data1 = pd.read_csv(train_path)
 #       print(self.train_data1.shape)
#         self.data = train_data.to_numpy()
        '''self.train_data = (self.train_data1.iloc[:19950,:]).to_numpy()
        print(self.train_data.shape)
#         valid = self.train_data1.iloc[19950:,1:]
#         labels_valid = self.train_data1.iloc[19950:,0]
#         print("shape ",labels_valid.shape)
        self.validation_data = (self.train_data1.iloc[19950:,1:]).to_numpy()
        labels_valid = self.train_data1.iloc[19950:,0].to_numpy()
        print(self.validation_data.shape)
#         test_labels = train_data1.iloc[19900:,-1]
        accuracy = self.predict_values_for_validation(self.train_data, self.validation_data, self.k  , distance_function = 'euclidean') 
        print (accuracy_score(labels_valid, accuracy))'''


# In[ ]:





# In[ ]:



'''def main():
    knn = KNNClassifier()
    knn.train("/home/ubuntu/Desktop/smai/Datasets/q1/train.csv")
    predictions = knn.predict("/home/ubuntu/Desktop/smai/Datasets/q1/test.csv")
#    print(predictions)
    test_labels = list()
    with open("/home/ubuntu/Desktop/smai/Datasets/q1/test_labels.csv") as f:
        for line in f:
            test_labels.append(line.strip())
#     print(test_labels[:10])
    count = 0
#     for i in range(len(test_labels)):
#         if test_labels[i] != predictions[i]:
#             count += 1
#     print(count)
    print (accuracy_score(test_labels, predictions))
    
    
    
    
    
    

if __name__ == '__main__':
    main()'''


# In[ ]:





# In[ ]:




