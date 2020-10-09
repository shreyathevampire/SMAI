#!/usr/bin/env python
# coding: utf-8

# In[4]:


from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
# print(mean_squared_error(y_test, y_pred))


# In[68]:



class AirFoil():
    def __init__(self):
        train_data = np.empty(2)
        labels = np.empty(1)
        theta = np.empty(1)

    def calculate_means(self,data):
        means = [0 for row in range(len(data[0]))]
        for i in range(len(data[0])):
            values = [x[i] for x in data]
            means[i] = sum(values)/(float(len(data)))

        return means

  
    def calculate_std(self,data, means):
        stddev = [0 for row in range(len(data[0]))]
        for i in range(len(data[0])):
            values = [pow(x[i]-means[i],2) for x in data]
            stddev[i] = math.sqrt((sum(values)/ float(len(data)-1)))
        return stddev

    def standardize_data(self,data,means,stddev):
        for row in data:
            for i in range(len(data[0])):
                row[i] = (row[i] - means[i])/stddev[i]
        return data

    def fit_values(self, X, thetas, y_train, alpha, max_iterations):
        X_transposed = X
        # max_iterations = 5000
        # alpha = 0.005
        threshold = 0.0001
        for i in range(max_iterations):
            diff = np.dot(X_transposed, thetas) - y_train
            cost = sum(diff**2)
            gradient = np.dot(diff, X_transposed)
            # print(gradient)
            value = (alpha*gradient)/float(len(X_transposed))
            # print(value)
            thetas = thetas - value

            # print(i," ",thetas)
        return thetas




    def train(self, path):
        data = pd.read_csv(path,header=None)
        labels_ = data.iloc[:,-1]
        data = data.iloc[:,:-1]
#         print(data.shape)
        self.train_data = data.to_numpy()
        self.labels = labels_.to_numpy()
        means = self.calculate_means(self.train_data)
        stddev = self.calculate_std(self.train_data,means)
        self.train_data = self.standardize_data(self.train_data,means,stddev)
#         print(self.train_data)
        #add a column of ones in the input dataset
        one = np.ones(len(self.train_data))
        new_input = np.concatenate((one[:, np.newaxis], self.train_data), axis=1)
        self.train_data = new_input
#         print("no of columns = " ,len(new_input[0]))
        '''thetas = np.ones(len(new_input[0]))
        alpha = 0.005
        max_iterations = 5000
        self.theta = self.fit_values(new_input, thetas, labels, alpha, max_iterations)
        print(self.theta)'''
        X_train = self.train_data[:800,:]
        y_train = self.labels[:800]
        X_test = self.train_data[800:,:]
        y_test = self.labels[800:]
        thetas = np.ones(len(new_input[0]))
        self.theta = self.fit_values(X_train, thetas, y_train, 0.005, 1000)
        y_pred = self.predict_values(X_test)
#                 print("predicted values")
#                 print(y_pred)
        value = mean_squared_error(y_test, y_pred)
        print(value)
        print(r2_score(y_test, y_pred))
        '''iterations = [10,50,100,500]
        alpha = [0.01,0.005]
        mse_error = []
        for i in alpha:
            thetas = np.ones(len(new_input[0]))
            mse_error_iter = []
            for j in iterations:
                
                self.theta = self.fit_values(X_train, thetas, y_train, i, j)
#                 print(self.theta)
                y_pred = self.predict_values(X_test)
#                 print("predicted values")
#                 print(y_pred)
                value = mean_squared_error(y_test, y_pred)
                print(value)
                print(r2_score(y_test, y_pred))
                mse_error_iter.append(value)
            mse_error.append(mse_error_iter)
        print("MSE = ",mse_error)
#         y_pred = self.predict_values(X_test)'''
        print("MSE = ",mean_squared_error(y_test, y_pred))



    def predict_values(self,X):
        # X_test = pd.read_csv(path, header = None)
        # X = X_test
        #add a column of ones in the input dataset
#         print("no of cols in test data = ", len(X[0]))
        # one = np.ones(len(X))

        # std_data_1 = standardize_data(X,means,stddev)
        # new_input = np.concatenate((one[:, np.newaxis], std_data_1), axis=1)
        # print(len(std_data_1[0]))
        # print(len(theta))
        y_pred = np.dot(X, self.theta)
        
        return y_pred

    def predict(self, path):
        X_test = pd.read_csv(path, header = None)
        X = X_test
        #add a column of ones in the input dataset
#         print("no of cols = ", len(X[0]))
        one = np.ones(len(X))
        means = self.calculate_means(self.train_data)
        stddev = self.calculate_std(self.train_data,means)
        std_data_1 = self.standardize_data(X,means,stddev)
        new_input = np.concatenate((one[:, np.newaxis], std_data_1), axis=1)
#         print(len(std_data_1[0]))
#         print(len(theta))
        y_pred = np.dot(new_input, self.theta)
        return y_pred



# In[69]:


af = AirFoil()
values = pd.read_csv("/home/ubuntu/Desktop/sem2/smai/assignment2/Datasets/Question-3/airfoil/airfoil.csv", header = None)
print(values.shape)
af.train("/home/ubuntu/Desktop/sem2/smai/assignment2/Datasets/Question-3/airfoil/airfoil.csv")
#y_pred = af.predict("/home/ubuntu/Desktop/sem2/smai/assignment2/Datasets/Question-3/airfoil/airfoil_test.csv")
# print("MSE = ",mean_squared_error(y_test, y_pred))

