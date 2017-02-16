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

import numpy as np
from numpy import linalg

import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn import cross_validation,grid_search, svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB

import matplotlib.pyplot as plt
import pandas as pd
import re
import csv
import time

import nose


class reviews():
    def __init__(self,restaurant,reviews,source):
        self.restaurant = restaurant
        self.reviews = reviews
        self.source = source
    
    def token (self,delim='\s+'):        
        tokenizer = RegexpTokenizer(delim, gaps=True)          
        self.reviews = [tokenizer.tokenize(review) for review in self.reviews]
    
    def punct_lowercase(self):
        '''Removes punctuation, converts to lowecase, and tokenizes words'''
        lower_case = [[re.sub("[^a-zA-Z']", "", word.lower()) for word in review] for review in self.reviews]
        self.reviews = lower_case
    
    def pos_tag(self,pos,keep_words,pos_t):
        '''Tag words, remove specific parts of speech from a given list, keep any given words, and assign pos tag or not'''

        item_pos = [nltk.pos_tag(self.reviews[i]) for i in xrange(len(self.reviews))]

        item_noun_adj=[]

        for review in item_pos:
            rev_noun_adj=[]
            for item in review:
                if item[1] in pos or item[0] in keep_words:
                    if pos_t==True: #POS tag included
                        rev_noun_adj.append(item)
                    else: #POS tag not included
                        rev_noun_adj.append(item[0])
            item_noun_adj.append(rev_noun_adj)
        self.reviews = item_noun_adj
    def noun_phrase(self):
        ''' Extracts 4 different noun phrases and returns a list of the noun phrases 
        in the order NOT|blank, adj, noun for the given corpus:

            First: adjective noun
            Second: noun verb adjective
            Third: NOT adjective noun
            Fourth: noun NOT adjective

        '''
        stemmer = SnowballStemmer("english")

        #Regex rules
        chunk_rule1 = ChunkRule("<JJ|JJS|JJR><NN|NNS>", "Chunk adjective noun")
        chunk_rule2 = ChunkRule("<NN|NNS><VB|VBD|VBG|VBN|VBP|VBZ><JJ|JJS|JJR>", "Chunk noun verb adjective")

        chunk_rule3 = ChunkRule("<NOT><.*><JJ|JJS|JJR><NN|NNS>", "Chunk NOT adjective noun")
        chunk_rule4 = ChunkRule("<NN|NNS><.*><NOT><.*><JJ|JJS|JJR>", "Chunk noun NOT adjective")

        nouns = ['NN', 'NNS']
        adjs = ['JJ', 'JJS', 'JJR']

        noun_phrases=[]


        for review in self.reviews:
            noun_phrase = []

            #Convert negation verbs and the word "'not' to 'NOT' tag
            review_chg = []
            for word in review:
                if "n't" in word[0] or word==('not', 'RB'):
                    review_chg.append(('not', 'NOT'))
                else:
                    review_chg.append(word)

            neg_index = [i for i in xrange(len(review_chg)) if review_chg[i]==('not', 'NOT')]

            if neg_index == []: #no negations
                chunk_parser = RegexpChunkParser([chunk_rule1,chunk_rule2], chunk_label='NP')
                chunked_text = chunk_parser.parse(review_chg)

                #extract only nouns and adjectives
                #http://streamhacker.com/2009/02/23/chunk-extraction-with-nltk/
                for subtree in chunked_text.subtrees(filter=lambda t: t.label() == 'NP'):
                    noun = stemmer.stem([n[0] for n in subtree.leaves() if n[1] in nouns][0])
                    adj = stemmer.stem([a[0] for a in subtree.leaves() if a[1] in adjs][0])

                    noun_phrase.append(adj+"_"+noun)
            else:

                chunk_parser = RegexpChunkParser([chunk_rule4,chunk_rule3,chunk_rule2,chunk_rule1], chunk_label='NP')
                chunked_text = chunk_parser.parse(review_chg)

                for subtree in chunked_text.subtrees(filter=lambda t: t.label() == 'NP'):
                    noun = stemmer.stem([n[0] for n in subtree.leaves() if n[1] in nouns][0])
                    adj = stemmer.stem([a[0] for a in subtree.leaves() if a[1] in adjs][0])

                    if [nt[0] for nt in subtree.leaves() if nt[1]=='NOT'] != []:
                        noun_phrase.append('not'+"_"+adj+"_"+noun)
                    else:                
                        noun_phrase.append(adj+"_"+noun)           

            noun_phrases.append(noun_phrase)

        self.reviews = noun_phrases
    def remove_stop(self,stop_words):
        item_lst=[]

        for review in self.reviews:
            rev_lst=[]
            for item in review:
                if item not in stop_words:
                    rev_lst.append(item)
            item_lst.append(rev_lst)


        self.reviews = item_lst
    
    def mispelled_remove(self):
        item_lst=[]

        for review in self.reviews:
            rev_lst=[]
            for item in review:
                if wn.synsets(item)!=[]: #wordnet returns an empty list for mispelled words
                    rev_lst.append(item)
            item_lst.append(rev_lst)

        self.reviews =  item_lst
    def SnowballStem(self):
        '''Stems words based on the Snowball Stemmer'''

        stemmer = SnowballStemmer("english")
        stemmed=[]

        for review in self.reviews:
            rev_lst=[]
            for item in review:
                rev_lst.append(stemmer.stem(item))
            stemmed.append(rev_lst)

        self.reviews = stemmed
    def remove_dup(self):
        '''Remove consecutive duplicate words. For example, 'love love love' would conver to a single 'love' '''
        t0 = time.time()
        dup_removed=[]

        for review in self.reviews:
            rev_lst=[]
            for i in xrange(len(review)-1):
                if i==0:
                    rev_lst.append(review[0])
                else:
                    if rev_lst[len(rev_lst)-2]!=review[i]:
                        rev_lst.append(review[i])
            dup_removed.append(rev_lst)


        self.reviews = dup_removed
