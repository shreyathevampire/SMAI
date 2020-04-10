#!/usr/bin/env python
# coding: utf-8

# In[7]:


#!/usr/bin/env python
# coding: utf-8

# In[89]:


import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from collections import Counter 
import statistics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from pprint import pprint
from sklearn.metrics import r2_score


# In[ ]:





# In[ ]:





# In[104]:


class DecisionTree():
    treee = {}
    def __init__(self, count = 0, 
                max_depth = 2,
                min_samples =2):
        print("")

    def cal_purity_of_data(self,data):
        label = data.iloc[:,-1]
        count = label.unique().tolist()
        if len(count) == 1:
            return True
        else :
            return False
        
    def drop_columns(self,data):
        data.drop('Id', axis=1, inplace =True)
        data.drop('Alley', axis=1, inplace =True)
        data.drop('PoolQC', axis=1, inplace =True)
        data.drop('Fence', axis=1, inplace =True)
        data.drop('MiscFeature', axis=1, inplace =True)
        return data
        
        
        
    def handle_missing_values(self, data):
        data = self.drop_columns(data)
        global fill_col
        fill_col =['YrSold','MoSold','MiscVal','PoolArea','ScreenPorch','3SsnPorch','EnclosedPorch','OpenPorchSF',
                 'WoodDeckSF','GarageArea','GarageCars','GarageYrBlt','Fireplaces','TotRmsAbvGrd','Kitchen','Bedroom',
                 'LotFrontage','LotArea','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF',
                 '1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath']
        for i in data.columns:
            if i in fill_col:
                data[i].fillna(data[i].mean(), inplace =True)
            else:
                  data[i].fillna(data[i].mode()[0], inplace = True)
        return fill_col,data
        
    def clean_data(self, data):
        fill_col,data = self.handle_missing_values(data)
        return fill_col,data
    
    def cal_words(self,fill_col, data):
        dictionary = []
        for i in data.columns:
            if ( i in fill_col or isinstance(data.iloc[0][i], str)):
                count = data[i].unique().tolist()
                dictionary.append(count)
            else:
                count = data[i].unique()
                count = sorted(count)
                lists = [(a+b)/2 for a,b in zip(count[::2], count[1::2])]
                dictionary.append(lists)
        return dictionary
        
    def train(self, path):
        data_t = pd.read_csv(path)[:]
        fill_col, data_t = self.clean_data(data_t)
        dictionary_of_unique_words = self.cal_words(fill_col,data_t)
        self.init_idx_dict(data_t.columns)
        
        self.treee = self.decision_tree_algo(data_t,0,3,3, dictionary_of_unique_words)

    def cal_mean(self,data):
        answer = 0
        if len(data) == 0:
            return answer
        else :
            answer = np.mean(data)
        return answer
    

    def split_data(self,data, column, split_at_value, mask = 1):
        if mask == 1:
            data_below_value = data[data.iloc[:,column] <= split_at_value]
            data_above_value = data[data.iloc[:,column] > split_at_value]
            return data_below_value, data_above_value
        else:
            data_below_value = data[data.iloc[:,column] == split_at_value]
            data_above_value = data[data.iloc[:,column] != split_at_value]
        return data_below_value, data_above_value
    
    def cal_mse(self,data):
        res= 0
        label_values = (data.iloc[:,-1]).tolist()
        if len(label_values) == 0:
            return 0
        avg = statistics.mean(label_values)
        sum = 0
        for i in range(len(label_values)):
            sum += (label_values[i] - avg)**2
        res = sum/len(label_values) 
        return res
    
    
    def cal_entropy_at_each_split(self,data,feature, value_at_feature):
        total_length = data.shape[0]
        if index_dictionary[feature] not in fill_col:
            data_below_value, data_above_value = self.split_data(data,feature,value_at_feature, 0)
        else:
            data_below_value, data_above_value = self.split_data(data,feature,value_at_feature, 1)
        prob = 0
        prob += (len(data_below_value)/total_length)*self.cal_mse(data_below_value)
        prob += (len(data_above_value)/total_length)*self.cal_mse(data_above_value)
        return prob,(data_below_value, data_above_value)

    
    


    def cal_entropy(self,data,dictionary):
        values_of_each_row = []
        for i in range(len(dictionary)-1):
            flag = True
            row = dictionary[i]
            mse_at_each_split = []
            for j in range(len(row)):
                temp_data = data
                entropy_at_each_split = self.cal_entropy_at_each_split(temp_data,i,row[j])[0]
                if flag == True or entropy_at_each_split < min_entropy :
                    min_entropy = entropy_at_each_split
                    feature = i
                    value_at_feature = row[j]
                    flag = False
            values_of_each_row.append((min_entropy,feature,value_at_feature))
        return min(values_of_each_row)[1], min(values_of_each_row)[2]
  
    
    def init_idx_dict(self, list_of_columns):
        global index_dictionary
        index_dictionary={}
        for i in range(len(list_of_columns)):
            index_dictionary[i] = list_of_columns[i]
            

    def decision_tree_algo(self,data, count, max_depth , min_samples , dictionary):
        rows = data.shape[0]
        if self.cal_purity_of_data(data) or rows <= min_samples or count == max_depth :
            lists = data.iloc[:,-1].tolist()
            answer = self.cal_mean(lists)
            return answer
        count = count + 1
        col_no, value_for_split = self.cal_entropy(data,dictionary)#
        values = self.cal_entropy_at_each_split(data, col_no, value_for_split)[1]
        data_below_value = values[0]
        data_above_value = values[1]
        list_of_columns = data.columns
        
        
        if index_dictionary[col_no] not in fill_col:
            question_to_ask = "{} == {}".format(col_no,value_for_split)
        else:
            question_to_ask = "{} <= {}".format(col_no, value_for_split)

        sub_tree = {question_to_ask : []}
        true_value = self.decision_tree_algo(data_below_value,count,max_depth,min_samples,dictionary)
        false_value = self.decision_tree_algo(data_above_value, count, max_depth, min_samples, dictionary)
        if true_value == false_value:
            sub_tree[question_to_ask].append(true_value)
        else:
            sub_tree[question_to_ask].append(true_value)
            sub_tree[question_to_ask].append(false_value)

        return sub_tree

    def testing_values(self,treee,test_data_point ):

        questions = list(self.treee.keys())[0]
        column_no, comparison_op, value_at_split = questions.split(' ')
        var = int(column_no)
        if comparison_op == "<=": #numerical data
            if test_data_point[var] <= float(value_at_split):
                solution = self.treee[questions][0]
            else : 
                solution = self.treee[questions][1]
        else: #categorical 
            if test_data_point[var] == value_at_split:
                solution = self.treee[questions][0]
            else : 
                solution = self.treee[questions][1]
                
        if not isinstance(solution,dict):
            return solution
        else:
            self.treee = solution
        return self.testing_values(self.treee, test_data_point)

    def predict(self, test_data_path):
        
        y_pred = []
        test_data = pd.read_csv(test_data_path)
        
        fill_col, test_data = self.clean_data(test_data)
        
        for i in range(test_data.shape[0]):

            test_data_point = test_data.iloc[i,:].to_numpy()
            y_pred.append(self.testing_values(self.treee,test_data_point))
        return y_pred


# In[108]:



