
# coding: utf-8

# In[31]:

import nltk
from nltk.corpus import stopwords
from nltk.collocations import *
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.metrics.association import QuadgramAssocMeasures
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
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
    
    def posTag_Filt(self,posKeep = ['NN','NNS','JJ','JJR','JJS','RB','RBR','RBS','VB','VBD','VBG','VBN','VBP','VBZ'],wordKeep = []):
        '''Tag words, remove specific parts of speech from a given list, and keep any given words'''

        self.token = [word for word, pos in nltk.pos_tag(self.token) if (pos in posKeep or word in wordKeep)]       
    
    def nounPhrase_Filt(self,chunks = [ChunkRule("<JJ|JJS|JJR><NN|NNS>", "Chunk adj. noun")]):
        '''Keep noun phrases based on a list of regex chunk rules'''
        
        chunkParser = RegexpChunkParser(chunks, chunk_label='NP')
        chunked_text = chunkParser.parse(nltk.pos_tag(self.token))
    
    def sortToken(self):
        '''Sort tokens. Used to better string distance calculations'''
        
        self.token.sort()
        
    def removeMisspelled(self):
        '''Remove misspelled words'''
        
        self.token = [word for word in self.token if wn.synsets(word)!=[]]
    
    def lemmatizer(self,lemmer = WordNetLemmatizer()):
        '''Lemmatize words'''
        
        self.token = [lemmer.lemmatize(word) for word in self.token]
        
    def stemmer(self,stemmer = SnowballStemmer("english")):
        '''Stems word, default Snowball Stemmer'''

        self.token = [stemmer.stem(word) for word in self.token]
        


# In[34]:

test_text = reviews('Skoobs, is a HAPPY dog dogs!!!!')
test_text.token


# In[35]:

test_text.sortToken()
test_text.token


# In[36]:

test_text.lowerCase()
test_text.token


# In[37]:

test_text.removePunct()
test_text.token


# In[82]:

test_text.removeStop()
test_text.token


# In[77]:

test_text.removeMisspelled()
test_text.token


# In[83]:

test_text.stemmer()
test_text.token


# In[23]:

test_text.posTag_Filt(['NN','NNS'])
test_text.token


# In[38]:

test_text.lemmatizer()
test_text.token


# In[55]:

rule1 = ChunkRule("<JJ|JJS|JJR><NN|NNS>",'Adj. Noun')
chunkParser = RegexpChunkParser([rule1], chunk_label='NP')
chunked_text = chunkParser.parse(nltk.pos_tag(['this','is','not','a','good','idea']))
        
#chunk_rule1 = ChunkRule("<JJ|JJS|JJR><NN|NNS>", "Chunk adjective noun")

