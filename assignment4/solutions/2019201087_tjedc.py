#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
# from google.colab.patches import cv2_imshow
import string
import os
import cv2
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from random import random
from sklearn.preprocessing import MinMaxScaler
#from skimage.transform import  downscale_local_mean
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
from keras.models import Sequential
from keras import layers
from keras import models
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import to_categorical
import pandas as pd


# In[2]:


path = '/home/ubuntu/Desktop/desktop/sem2/smai/assignment4/tom_n_jerry/'
normalize = Normalizer()
def generate_data(traindf):
    data = []
    normalizer = Normalizer()
    for i in range(len(traindf)):
#         print(i)
        img_path = path + str(traindf.iloc[i][0])
#         print(img_path)
        gray_image = cv2.imread(img_path,0)
        # print(gray_image)
        scale_percent = 50 # percent of original size
        # width = int(gray_image.shape[1] * scale_percent / 100)
        # height = int(gray_image.shape[0] * scale_percent / 100)
        dim = (64,64)
        # resize image
        resized = cv2.resize(gray_image, dim, interpolation = cv2.INTER_AREA)

        resized = normalizer.fit_transform(resized)
        data.append(resized)

    data = np.array(data)
    print(data.shape)
    return data


# In[5]:


class Cartoon:
    model = None
    def train(self,path):
        traindf=pd.read_csv(path,dtype=str)
        traindf["image_file"]=traindf["image_file"].apply(append_ext)
        X_train = generate_data(traindf)
        train_labels = to_categorical(traindf['emotion'])
        train_labels.shape
        self.model = Sequential()
        self.model.add(layers.Conv2D(128, (3, 3), padding='same',
                         input_shape=(64,64,1)))
        self.model.add(layers.Activation('relu'))
        self.model.add(layers.pooling.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(layers.Conv2D(128, (3, 3)))
        self.model.add(layers.Activation('relu'))
        self.model.add(layers.pooling.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(layers.Conv2D(128, (3, 3), padding='same'))
        self.model.add(layers.Activation('relu'))
        self.model.add(layers.pooling.MaxPooling2D(pool_size=(2, 2)))
        # model.add(layers.Conv2D(64, (3, 3)))
        # model.add(layers.Activation('relu'))
        # model.add(layers.pooling.MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128))
        # model.add(layers.Activation('relu'))
        # model.add(Dropout(0.5))
        self.model.add(Dense(5, activation='softmax'))
        self.model.compile(optimizers.Adam(learning_rate=0.001),loss="categorical_crossentropy",metrics=["accuracy"])
        X = X_train.reshape([-1,64,64,1])
        self.model.fit(x=X, y= train_labels, batch_size=64, epochs= 1)
        
        
        
        
        
    def predict(self,path):
        testdf = pd.read_csv(path, dtype= str)
        testdf["image_file"] = testdf["image_file"].apply(append_ext_test)
        X_test = generate_data(testdf)
#         y_test = pd.read_csv('/home/ubuntu/Desktop/desktop/sem2/smai/assignment4/tom_n_jerry/test_org_labels.csv', dtype= str)
#         y_test = y_test['emotion']
#         print(y_test.head())
#         y_test = np.array(y_test)
        Xtest = X_test.reshape([-1,64,64,1])
        preds = self.model.predict(Xtest)
        print(preds)
        pred_list = np.argmax(preds, axis = -1)
        # test_labels = np.argmax(np.array(preds))
        print(pred_list)
#         print(y_test)
#         print(y_test[0]," ",type(y_test[0]))
#         ytest = [int(i) for i in y_test]
#         print(ytest)
        return pred_list
        # y_test = np.array(y_test)
        # print(y_test[1])
#         from sklearn.metrics import accuracy_score, f1_score
#         print(accuracy_score(ytest, pred_list))
#         print(f1_score(ytest,pred_list,average='macro'))
        


# In[ ]:





# In[ ]:





# In[ ]:





# In[6]:


def append_ext(fn):
    return "ntrain/"+fn+".jpg"

def append_ext_test(fn):
    return "ntest/"+fn+".jpg"



ct = Cartoon()
ct.train('/home/ubuntu/Desktop/desktop/sem2/smai/assignment4/tom_n_jerry/train.csv')
predictions = ct.predict('/home/ubuntu/Desktop/desktop/sem2/smai/assignment4/tom_n_jerry/test.csv')
df = pd.DataFrame({'labels': predictions})
df.to_csv('/home/ubuntu/Desktop/desktop/sem2/smai/assignment4/tom_n_jerry/temp_submission.csv', index=True)
print(df.shape)


