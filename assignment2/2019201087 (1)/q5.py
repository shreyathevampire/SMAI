#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries

# In[6]:


import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import re
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
nltk.download('stopwords')
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


# # AuthorClassifier class

# In[7]:


class AuthorClassifier:
    
    def __init__(self):
        self.svc = SVC(kernel = "linear", gamma = 'auto')
        self.termidf = TfidfVectorizer(lowercase=False)
    
    def preprocess_data(self,train_data):
        stopword = stopwords.words('english')
        corpus = np.empty(len(train_data), dtype=object)
        ps = PorterStemmer()
        # print(data)
        print("=====================")
        for i in range(len(train_data)):
            # print(i)
#             print(train_data[i])
            text = train_data[i]
#             print("text type" ,type(text)," ",text)
            # text = np.array_str(text)
            # print(type(text))

            # keeping only words in the review text
            review = re.sub('[^a-zA-Z]', ' ', text)
            # #convert everything to lower  case
            review = review.lower()
            # #split it into words
            review = review.split()
            # #remove stopwords and do stemming
            review = [ps.stem(word) for word in review if not word in stopword]
            # #the words are joined together to form a sentence
            review = ' '.join(review)
            # #the cleaned text is appended to corpus
            corpus[i] = review
            # corpus.append(review)
        return corpus
 
    
    
    
    
    def train(self,path):
        data1 = pd.read_csv(path)
#         print(data1)
#         print(data1.isnull())
        data = data1.to_numpy()
#         print(data[0])
#         print(pd.DataFrame(data[:,2]).isnull())
        
        
        train_labels = data[:,2]
        train_data = data[:,1]
#         test_labels = data[15000:,-1]
#         test_data = data[15000:,:-1]
        print(train_data.shape)
        # print(train_data)
        corpus = []

        corpus_train_data = self.preprocess_data(train_data)
#         corpus_test_data = self.preprocess_data(test_data)
        # print(type(corpus_train_data[0]), "shape of train_data ", corpus_train_data.shape)
        # print(type(corpus_test_data))


        
        tfidf_train_data = self.termidf.fit_transform(corpus_train_data)
        print(type(tfidf_train_data))
#         tfidf_test_data = termidf.transform(corpus_test_data)
        # print(tfidf_test_data)
        # print("tfidf shape",tfidf_train_data, tfidf_test_data)
#         svc = SVC(kernel = "linear", gamma = 'auto')
        self.svc.fit(tfidf_train_data, train_labels)

#         y_pred = svc.predict(tfidf_test_data)
        # print(y_pred)
#         print("SVC using TFIDF",accuracy_score(test_labels,y_pred))

#         # print(tfidf)
#         svc = SVC(kernel = "poly")
#         svc.fit(tfidf_train_data, train_labels)

#         y_pred = svc.predict(tfidf_test_data)
#         # print(y_pred)
#         print("SVC using TFIDF",accuracy_score(test_labels,y_pred))'''
        
    def predict(self, path):
        tfidf_test_data = pd.read_csv(path)
        print("no of elements in test data = ", tfidf_test_data.shape)
        tfidf_test_data = tfidf_test_data.to_numpy()
        print(type(tfidf_test_data))
        tfidf_test_data = tfidf_test_data[:,1]
#         print(tfidf_test_data)
        corpus_test_data = self.preprocess_data(tfidf_test_data)
        print(type(corpus_test_data))
        tfidf_test_data = self.termidf.transform(corpus_test_data)
        y_pred = self.svc.predict(tfidf_test_data)
        return y_pred
        
  


# In[ ]:


##predict method left


# In[12]:




acc = AuthorClassifier()
acc.train('/home/ubuntu/Desktop/sem2/smai/assignment2/Datasets/Question-5/Train.csv')
y_pred = acc.predict('/home/ubuntu/Desktop/sem2/smai/assignment2/Datasets/Question-5/test.csv')
test_labels = []
with open("/home/ubuntu/Desktop/sem2/smai/assignment2/Datasets/Question-5/test_labels.csv") as f:
    for line in f:
        test_labels.append(line.strip())
    f.close()
# print(test_labels)
print (accuracy_score(test_labels[1:], y_pred))
print(f1_score(test_labels[1:], y_pred, average='macro'))
print(confusion_matrix(test_labels[1:], y_pred))


# In[ ]:




