
# coding: utf-8

# In[78]:

import nltk
from nltk.corpus import stopwords
from nltk.collocations import *
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.metrics.association import QuadgramAssocMeasures
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import *
from nltk.corpus import wordnet as wn
from nltk.chunk import *
from nltk.chunk.util import *
from nltk.chunk.regexp import *
from nltk import Tree
import re

class reviews(object):
    def __init__(self,text,delim='\s+'):
        '''Tokenize words and keep original text'''
        
        self.orig_text = text
        
        tokenizer = RegexpTokenizer(delim, gaps=True) 
        self.token = tokenizer.tokenize(self.orig_text)
    
    def lowerCase(self):
        '''Converts tokenized works to lower case'''
        
        self.token = [word.lower() for word in self.token]
    
    #def lowerFirst_word(self):
     #   '''Convert the first word of a sentance to lower case'''
    
    def removePunct(self,reg = "[^a-zA-Z']"):
        '''Removes punctuation except apostrophes (default)'''
        
        self.token = [re.sub(reg, "", word) for word in self.token]
    
    def removeStop(self,stop_words = set(stopwords.words('english'))):
        '''Remove stop words'''
        
        self.token = [word for word in self.token if word not in stop_words]
    
    #Need to add pos tagging
    
    #Need to add noun phrases
    
    def removeMispelled(self):
        '''Remove misspelled words'''
        
        self.token = [word for word in self.token if wn.synsets(word)!=[]]
    
    def stemmer(self,stemmer = SnowballStemmer("english")):
        '''Stems word, default Snowball Stemmer'''

        self.token = [stemmer.stem(word) for word in self.token]
        


# In[79]:

test_text = reviews('Skoobs, is a HAPPY dog doggy!!!!')
test_text.token


# In[80]:

test_text.lowerCase()
test_text.token


# In[81]:

test_text.removePunct()
test_text.token


# In[82]:

test_text.removeStop()
test_text.token


# In[77]:

test_text.removeMispelled()
test_text.token


# In[83]:

test_text.stemmer()
test_text.token

