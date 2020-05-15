#!/usr/bin/env python
# coding: utf-8

# In[2]:


import glob
import errno


# In[4]:


import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import re
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics


# In[121]:



'''files = glob.glob(path)
labels = []

count = 0
for name in files:
    count += 1
corpus = np.empty(count,dtype=object)
i = 0
for name in files:
    data = []
    print(name)
    name_split = name.split('/')
    file_name = name_split[-1].split('.')
    name_file = file_name[0].split('_')
    labels.append(name_file[1])
    try:
        with open(name, 'rb') as f:
            text = f.read()
            corpus[i] = text
            i += 1
    except IOError as exc: #Not sure what error this is
        if exc.errno != errno.EISDIR:
            raise'''
            


# In[11]:


class Cluster:
    def __init__(self):
        max_iter = 500;
        

        
        
    def initialize_centroids_data(self, data, no_of_clusters):
        initialize_centroids = data[np.random.randint(data.shape[0]), : no_of_clusters]
        return initialize_centroids
    
    def KMeans(self, data, max_iterations, labels, no_of_clusters):
        self.max_iter = max_iterations
        initial_centroids = np.random.permutation(data.shape[0])[:no_of_clusters]
        data_centroids = data[initial_centroids]
        for i in range(30):
            dist_to_centroids = pairwise_distances(data, data_centroids, metric = 'euclidean')
            labels = np.argmin(dist_to_centroids, axis = 1)
            for i in range(no_of_clusters):
                table = []
                for j in range(len(labels)):
                    if labels[j] == i:
                        table.append(data[j][0])
                table = np.array(table)  
                new_img = table.reshape((table.shape[0]*table.shape[1]), table.shape[2])
                table = new_img
                data_centroids[i] = table.mean(axis = 0)
            
        return labels
        
        
        
        
    
    def preprocess_data(self,text):
        stopword = stopwords.words('english')
        ps = PorterStemmer()
        
        review = re.sub('[^a-zA-Z]', ' ', text)
        
        review = review.lower()
        
        review = review.split()
        
        review = [ps.stem(word) for word in review if not word in stopword]
        
        review = ' '.join(review)
        return review
        
    
    
    def cluster(self, path):
        files = glob.glob(path)
        
        labels = []

        count = 0
        for name in files:
            count += 1
        corpus = np.empty(count,dtype=object)
        i = 0
        filenames = []
        for name in files:
            data = []
            name_split = name.split('/')
            file_name = name_split[-1].split('.')
            filenames.append(name_split[-1])
            name_file = file_name[0].split('_')
            labels.append(name_file[1])
            try:
                with open(name, encoding="utf8", errors='ignore') as f:
                    text = f.read()
                    text = self.preprocess_data(text)
                    corpus[i] = text
                    i += 1
            except IOError as exc:
                if exc.errno != errno.EISDIR:
                    raise

        
            
        tf_idf_vectorizor = TfidfVectorizer(stop_words = 'english',
                             max_features = 20000)
        tf_idf = tf_idf_vectorizor.fit_transform(corpus)
        
        tfidf = tf_idf.todense()
        
        labels = np.asarray(labels)
        y_pred = self.KMeans(tfidf, 500, labels, 5)
        print(y_pred)
        
        
        print(metrics.fowlkes_mallows_score(labels, y_pred))
        
        
    
#         count = 0
#         for i in range(len(labels)):
#             if int(labels[i]) == y_pred[i]:
#                 count += 1
#         print("count of labels matching = ", count)
        
        
        
        
cl = Cluster()
cl.cluster('/home/ubuntu/Desktop/sem2/smai/assignment2/Datasets/Question-6/dataset/*.txt')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




