#!/usr/bin/env python
# coding: utf-8

# # SpacyNER for Project 3

# In[22]:


get_ipython().system('pip3 install nltk')


# In[121]:


import scispacy
import spacy

import en_ner_bc5cdr_md   
import en_ner_bionlp13cg_md
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import nltk 
import string 
import re
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

from spacy import displacy


# In[122]:


# use pandas to import csv file

df = pd.read_csv("metadata_April10_2020.csv")



print(df.head())



# ## Load Models 

# In[123]:


nlp_bc = spacy.load("en_ner_bc5cdr_md")
nlp_bio =spacy.load("en_ner_bionlp13cg_md")


# ## Methods to add enitity/value pairs 

# In[124]:


def add_bc(abstractList, pbmdidList):
    i=0
    table = {"pubmed_id":[], "Entity":[], "Class":[]}
    for doc in nlp_bc.pipe(abstractList):
        pubmed_id = pbmdidList[i]
        for x in doc.ents:
            table["pubmed_id"].append(pubmed_id)
            table ["Entity"].append(x.text)
            table ["Class"].append(x.label_)
        i+=1
    return table 


# In[125]:


def add_bio(abstractList, pbmdidList):
    i=0
    table = {"pubmed_id":[], "Entity":[], "Class":[]}
    for doc in nlp_bio.pipe(abstractList):
        pubmed_id = pbmdidList[i]
        for x in doc.ents:
            table["pubmed_id"].append(pubmed_id)
            table ["Entity"].append(x.text)
            table ["Class"].append(x.label_)
        i+=1
    return table 


# In[126]:


# read csv file

#df = pd.read_csv("NewData.csv")
# remove empty abstracts 
df = df.dropna(subset=['abstract'] )

#create Lists
pbmdidList = df['pubmed_id'].tolist()
abstractList = df['abstract'].tolist()

# Add all entity pairs to tables 

table = add_bc(abstractList, pbmdidList)

# table = add_bio(abstractList, pbmdidList)

#turn table into exportable csv

trans_df = pd.DataFrame(table)



trans_df.to_csv ("NewDrugDisease3.csv", index=False)


# In[ ]:




