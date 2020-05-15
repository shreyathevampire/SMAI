#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import emoji
from nltk.tokenize import RegexpTokenizer

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score


# In[10]:


from nltk.corpus import stopwords
stopwords = set(stopwords.words('english')) - {'no', 'nor', 'not'}

def remove_stopwords(text):
    return ' '.join([word.lower() for word in str(text).split() if word not in stopwords])


# In[11]:


def final_data_cleaning(train_data):
    train_data['cleanedText'] = train_data['noemoji'].apply(lambda text: remove_stopwords(text))
    train_data['text_lem'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]',' ',text)) for text in lis]) for lis in train_data['cleanedText']]
    return train_data


# In[12]:


def clean_text(df):
    for i in range(df.shape[0]):
        txt = df.iloc[i]['text']
        txt = txt.lower()
        txt = re.sub('https?://[A-Za-z0-9./]+','',txt)#replace URLs
        txt=re.sub(r'@[A-Z0-9a-z_:]+','',txt)#replace username-tags
        txt=re.sub(r'^[RT]+','',txt)
        txt = re.sub(r'[0-9]*','',txt) #replace numbers
        txt=re.sub(r'#', '',txt)#replace hashtags
        txt = re.sub(r'[^a-zA-Z]',' ',txt) #replace punctuations
        df.at[i,'text'] = txt
        
    '''replace emotions'''
    for i, rows in df.iterrows():
        val = df.iloc[i]['text']
        new_val = ""
        for l in range(len(val)):
            if val[l] not in emoji.UNICODE_EMOJI:
                new_val += val[l]
                
#         print(new_val)
        df.at[i,'noemoji'] = new_val
    
    return df


# In[15]:


class HateSpeechDetection:
    modelLR = None
    vect = None
    
    def train(self,path):
        train_set = pd.read_csv('/home/ubuntu/Desktop/desktop/sem2/smai/assignment4/hate_speech/train.csv')
        y_train = train_set['labels']
        print(y_train.shape)
        train_set.drop(['labels'], axis=1, inplace = True)
        train_data = clean_text(train_set)
        # print(train_data)
        training_data = final_data_cleaning(train_data)
        # print(training_data.columns)

        X_train = training_data['text_lem']
#         X_test = testing_data['text_lem']

        self.vect = TfidfVectorizer(ngram_range = (1,4)).fit(X_train)

        vect_transformed_X_train = self.vect.transform(X_train)
        
        self.modelLR = LogisticRegression(C=0.1).fit(vect_transformed_X_train,y_train)

#         predictionsLR = smodelLR.predict(vect_transformed_X_test)
#         print(sum(predictionsLR==1),f1_score(predictionsSVC,predictionsLR))
        # from sklearn.metrics import accuracy_score
        # print(accuracy_score(y_test,predictionsLR))
        # print(f1_score(y_test,predictionsLR))
        

        
    def predict(self, path):
        test_set = pd.read_csv(path)
        test_data = clean_text(test_set)
        testing_data = final_data_cleaning(test_data)
        X_test = testing_data['text_lem']
        vect_transformed_X_test = self.vect.transform(X_test)
        predictionsLR = self.modelLR.predict(vect_transformed_X_test)
        return predictionsLR
        


# In[ ]:





# In[17]:


# train_set = pd.read_csv('/home/ubuntu/Desktop/desktop/sem2/smai/assignment4/hate_speech/train.csv')



ht = HateSpeechDetection()
ht.train('/home/ubuntu/Desktop/desktop/sem2/smai/assignment4/hate_speech/train.csv')
predictionsLR = ht.predict('/home/ubuntu/Desktop/desktop/sem2/smai/assignment4/hate_speech/test.csv')
df = pd.DataFrame({'labels': predictionsLR})
df.to_csv('/home/ubuntu/Desktop/desktop/sem2/smai/assignment4/hate_speech/temp_submission.csv', index=True)
print(df.shape)


# In[ ]:





# In[ ]:




