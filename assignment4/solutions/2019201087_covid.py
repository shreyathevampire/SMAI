#!/usr/bin/env python
# coding: utf-8

# In[1]:



from rdkit import Chem 
#get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from rdkit.Chem import Descriptors
from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from csv import writer
from csv import reader
from csv import DictReader
from csv import DictWriter


# In[2]:


model = word2vec.Word2Vec.load('/home/ubuntu/Desktop/desktop/sem2/smai/assignment4/covid19/model_300dim.pkl')
def data_transforming(traindf):
    #Transforming SMILES to MOL
    traindf['mol'] = traindf['SMILES sequence'].apply(lambda x: Chem.MolFromSmiles(x))
    
    print('Molecular sentence:', mol2alt_sentence(traindf['mol'][1], radius=1))
    print('\nMolSentence object:', MolSentence(mol2alt_sentence(traindf['mol'][1], radius=1)))
    print('\nDfVec object:',DfVec(sentences2vec(MolSentence(mol2alt_sentence(traindf['mol'][1], radius=1)), model, unseen='UNK')))
    #Constructing sentences
    traindf['sentence'] = traindf.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], 1)), axis=1)

    #Extracting embeddings to a numpy.array
    #Note that we always should mark unseen='UNK' in sentence2vec() so that model is taught how to handle unknown substructures
    traindf['mol2vec'] = [DfVec(x) for x in sentences2vec(traindf['sentence'], model, unseen='UNK')]
    X = np.array([x.vec for x in traindf['mol2vec']])
    return X


# In[5]:


class Covid19:
    svm_classifier = None
    def train(self,path):
        traindf = pd.read_csv(path)
        y = traindf['Binding Affinity']
        traindf.drop(columns='Binding Affinity',inplace=True)
        X = data_transforming(traindf)
        self.svm_classifier = SVR(kernel = 'rbf', C=100, epsilon = 1)
        self.svm_classifier.fit(X, y)

    def predict(self,path):
        testdf = pd.read_csv(path)
        print(testdf.shape)
        y_test = testdf['Binding Affinity']

        testdf.drop(columns='Binding Affinity',inplace=True)
        X_test = data_transforming(testdf)
        prediction_svr = self.svm_classifier.predict(X_test)
        df = pd.DataFrame({'SMILES sequence':testdf.iloc[:,0], 'Binding Affinity':prediction_svr})
        return df


# In[ ]:





# In[4]:


cv = Covid19()
cv.train('/home/ubuntu/Desktop/desktop/sem2/smai/assignment4/covid19/train.csv')
df = cv.predict('/home/ubuntu/Desktop/desktop/sem2/smai/assignment4/covid19/submission.csv')
df.to_csv('/home/ubuntu/Desktop/desktop/sem2/smai/assignment4/covid19/temp_submission.csv', index=False)
print(df.shape)


# In[ ]:




