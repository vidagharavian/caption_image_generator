#!/usr/bin/env python
# coding: utf-8

# # Requirement

# please install all requirements before running project

# In[1]:


import os
from nltk.corpus import stopwords
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier


# in this function we can get our labels for train and test datasets

# In[2]:


def get_class_names(is_train:bool=True):
    if is_train:
        list_of_files=os.listdir('dataset/train/sentences/')
    else:
        list_of_files=os.listdir('dataset/test/sentences/') 
    return list_of_files
get_class_names()
get_class_names(False)


# we can get file name of each labels with In[10] but we should mention that if we want training dataset or testdataset

# In[3]:


def get_class_file_names(class_name,train_test):
        list_of_files=os.listdir(f'dataset/{train_test}/sentences/{class_name}/')
        return list_of_files
get_class_file_names('aeroplane','test')


# this function is for finding the most ferequent words in given text

# In[4]:


def clean_x(k,X_train):
        all_words=X_train.split(" ")
        import nltk
        nltk.download('stopwords')
        stop_words = stopwords.words('english')
        text = [word for word in all_words if word not in stop_words]
        text = [word for word in text if word not in ["a", "us", "c"]]
        text = ' '.join(text)
        import re
        remove_punctuation_regex = re.compile(
            r"[^A-Za-z ]")  # regex for all characters that are NOT A-Z, a-z and space " "
        text = re.sub(remove_punctuation_regex, "", text)
        from nltk import FreqDist
        all_words = text.split()  # list of all the words in your corpus
        fdist = FreqDist(all_words)  # a frequency distribution of words (word count over the corpus)
         # say you want to see the top 10,000 words
        top_k_words, _ = zip(*fdist.most_common(k))
        return top_k_words
clean_x(10,"Romeo and Juliet is a tragedy written by William Shakespeare early in his career about two young star-crossed lovers whose deaths ultimately reconcile their feuding families. It was among Shakespeare's most popular plays during his lifetime and, along with Hamlet, is one of his most frequently performed plays. Today, the title characters are regarded as archetypal young lovers.")


# we want to get most frequent words in training data sets and use them as features that our classification is 
# based on it

# In[5]:


def get_all_features():
    labels=get_class_names()
    all_file = ""
    for label in labels:
        file_names=get_class_file_names(label,'train')
        for file in file_names:
            file=open(f'dataset/train/sentences/{label}/{file}','r')
            r=file.read()
            all_file=all_file+r.lower()
    top_k_words=clean_x(400,all_file)
    return top_k_words
get_all_features()


# Now we should create dataset after feature extarction v1

# In[7]:


def create_x(test_train):
    features=list(get_all_features())
    labels=get_class_names()
    out_put=[]
    header=features
    for i,label in enumerate(labels):
        files=get_class_file_names(label,test_train)
        for file in files:
            row=[]
            file=open(f'dataset/{test_train}/sentences/{label}/{file}','r')
            text=file.read()
            for feature in features:
                row.append(text.count(feature))
            row.append(label)
            out_put.append(row)
    header.append('label')
    df = pd.DataFrame(out_put, columns=header)
    df.to_excel(f'{test_train}.xlsx', sheet_name='Sheet1', index=False)
    
create_x('train')
create_x('test')


# In[12]:



class Classifire():
    @staticmethod
    def similar(a, b):
        if a == 'woman' and b == 'women':
            return True
        if a == 'man' and b == 'men':
            return True
        return  a+'s' == b or a+'a'==b

    @classmethod
    def get_same_header(cls):
        headers=pd.read_excel('train.xlsx',sheet_name='Sheet1',engine='openpyxl').columns
        output={}
        for header in headers:
            keys=list(output.keys())
            output[header]=[]
            for key in keys:
                if cls.similar(header,key) or cls.similar(key,header):
                    output[key].append(header)
                    try:
                        del output[header]
                    except:
                        pass
        return output
    @classmethod
    def get_last_data(cls,test_train):
        data = pd.read_excel(f'{test_train}.xlsx',sheet_name='Sheet1',engine='openpyxl')
        y=data['label']
        x=data
        x=x.drop(columns=['label'])
        same_headers=cls.get_same_header()
        for key,value in same_headers.items():
            if len(value)!=0:
                data=x[value].values
                for i,item in enumerate(x[key]):
                    item+=sum(data[i])
                x=x.drop(columns=value)
        return x,y.values
    @classmethod
    def mlp_clf(cls):
        X_train,y_train=cls.get_last_data('train')
        X_test,y_test=cls.get_last_data('test')
        clf = MLPClassifier(random_state=1, max_iter=3000,activation='logistic').fit(X_train,y_train.ravel())
        print(clf.score(X_test, y_test.ravel()))
    @classmethod
    def random_forest(cls):
        X_train,y_train=cls.get_last_data('train')
        X_test,y_test=cls.get_last_data('test')
        clf = RandomForestClassifier(random_state=1,max_depth=30).fit(X_train,y_train.ravel())
        print(clf.score(X_test, y_test.ravel()))
        import joblib
        joblib.dump(clf, 'caption_model.pkl')
        return clf
clf=Classifire()
clf.mlp_clf()
random_forest_model =clf.random_forest()


# and we can load our trained model for using it after with this function.

# In[14]:


import joblib 
random_forest_joblib = joblib.load('caption_model.pkl')

