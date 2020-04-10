#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import Sequential
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import pandas as pd
import sys


# In[ ]:


class RegressionSeries:
    def readcsv(self,path):
        df = pd.read_csv(path, sep=';', 
                 parse_dates={'dt' : ['Date', 'Time']}, infer_datetime_format=True, 
                 low_memory=False, index_col='dt')
        return df
    
    
    def data_collection(self,df):
        active_values = []
        start_index = []
        end_index = []
        rows = df.shape[0]
        for i in range(rows):
            val = df.iloc[i,0]
            if val == '?' or val == 'nan':
                if len(active_values)-60 >= 0:
                    start_index.append(i-60)
                    end_index.append(i-1)
                else :
                    start_index.append(0)
                    if i == 0:
                        end_index.append(0)
                    else: 
                        end_index.append(i-1)
            else:
                active_values.append(val)
        return active_values, start_index, end_index
                
                
    def prepare_data(self, active_values):
        training = []
        labels = []
        for i in range(len(active_values)):
#             print(i)
            if i+60 < len(active_values):
                arr = active_values[i:i+60]
#                 print(len(arr))
                training.append(arr)
                labels.append(active_values[i+60])
            else : 
                break
        #print("trainging= ", len(training))
        training = np.asarray(training)
        #print("trining np = ", training.shape)
        labels = np.array(labels)
        return training, labels
        
        
    def train(self, train_data, labels):
        
        train_set = train_data
        train_labels = labels
#         print(train_set.shape)
#         print(train_labels.shape)
#         print(type(train_set[0]))
        self.model = Sequential()
        self.model.add(Dense(100, activation="tanh", input_dim=60))
        self.model.add(Dense(70, activation="relu"))
        self.model.add(Dense(30, activation="relu"))
        self.model.add(Dense(1))
        self.model.compile(optimizer="adam", loss="mse", metrics=['mse', 'mae', 'mape'])
        self.model.fit(train_set, train_labels, epochs=20, batch_size=1000)
        
        
    def predict_values(self, start_index, end_index, active_values):
    
        preds = []
        for i in range(len(start_index)):
#             print(i)
            x = active_values[start_index[i]: end_index[i]+1]
            x = np.array(x)
            x = x.reshape((1,60))
            val = self.model.predict(x)
            preds.append(val)

        return preds

        


# In[ ]:


lr = RegressionSeries()
df = lr.readcsv(sys.argv[1])
#print("data read into dataframe")
act_values, s_ind, e_ind = lr.data_collection(df)
#print(len(act_values))
#print(len(s_ind))
#print(len(e_ind))
#print("divided data into test and training set")
train_data, labels  = lr.prepare_data(act_values)
# print(len(train_data))
# print("train = ", train_data.shape)
#print("prepared data into train_data")
lr.train(train_data, labels)
preds = lr.predict_values(s_ind,e_ind, act_values)
print(preds)


# In[ ]:


# print(preds)

