
# coding: utf-8

# In[1]:

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


# In[2]:

cd C:\Users\jmn251pitt\Documents\CBrands\Store Profile


# In[3]:

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


# In[46]:

from reviews import punct_lowercase
def test_ABCD():
    assert punct_lowercase(['ABCD']) == ['abcd']


# In[48]:

nosetests test_ABCD


# In[4]:

#import review file
data = pd.read_csv('Store Review.csv')
#rose_reviews = data[data['Restaurant']=='Rosebud Prime']['Review']
rose_reviews = data[data['Restaurant']=='RosebudPrime']['Review']
#rose_source = data[data['Restaurant']=='Rosebud Prime']['Source']
rose_source = data[data['Restaurant']=='RosebudPrime']['Source']
rosebudprime = reviews('RosebudPrime',rose_reviews,rose_source)


# In[5]:

rosebudprime.token()
rosebudprime.punct_lowercase()


# In[6]:

rosebudprime.reviews


# In[34]:

rosebudprime.mispelled_remove()


# In[36]:

rosebudprime.SnowballStem()


# In[27]:

pos = ('NN','NNS', #nouns
       'JJ','JJR', 'JJS', #adjectives
       'RB','RBR','RBS', #adverbs
       'VB','VBD','VBG','VBN','VBP','VBZ') #verbs

keep_words = ('but','while')

rosebudprime.pos_tag(pos,keep_words,True)


# In[38]:

#Remove keep words from stopwords list
stop_words = [word for word in stopwords.words("english") if word not in ['but','while','not','however','so']]


# In[39]:

rosebudprime.remove_stop(stop_words)


# In[41]:

#rosebudprime.reviews


# In[1046]:

dup_removed[0]


# ###### Combine adj, nouns, and noun phrases

# In[1047]:

final_terms = []
for i in xrange(len(dup_removed)):
    final_terms.append(dup_removed[i] + n_p[i])


# In[1048]:

final_terms[0]


# ### Convert to Document-Term Matrix

# In[1049]:

#First convert to dictionary with the review text converted back to a string
#This is needed for the Frequency counter (CountVetorizer) below
index=0

doc_term_dict={}

for i in xrange(len(final_terms)):
    doc_term_dict[index] = ' '.join(final_terms[index])
    
    index+=1    


# In[1050]:

doc_term_dict[0]


# In[1051]:

t0 = time.time()
#count = CountVectorizer(ngram_range=(1,2)) #unigras and bigrams
count = CountVectorizer() #unigrams
doc_term = count.fit_transform(doc_term_dict.values())
t1 = time.time()
print t1-t0


# In[1052]:

#Extract terms in the order of the document term matrix into a list

terms = [None] * len(count.vocabulary_.keys()) #empty list with size equal to the amount of terms

for term in count.vocabulary_.keys():
    terms[count.vocabulary_[term]]=term


# In[1053]:

len(terms)


# In[1054]:

shape(doc_term)


# ### Remove words with frequencies <=2

# In[1055]:

#Term frequency totals
term_freq = doc_term.sum(axis=0)


# In[1056]:

term_freq.shape


# In[1057]:

doc_term.shape


# In[1058]:

#Check how many terms have more than two terms
term_freq1 = term_freq>2

term_freq1.sum()


# In[1059]:

#Find column indexs which are <=2
term_index = np.where(term_freq>2)[1]
term_index = np.resize(term_index,shape(term_index)[1])
#print term_index


# In[1060]:

shape(term_index)


# In[1061]:

#Remove those columns from the doc_term matrix and the term list
terms_GT2 = [terms[i] for i in term_index]
doc_term_GT2 = doc_term[:,term_index]


# In[1062]:

#Check shapes
print shape(terms_GT2)
print shape(doc_term_GT2)


# ## Term Weights

# ### TFIDF

# In[ ]:

#http://stanford.edu/~rjweiss/public_html/IRiSS2013/text2/notebooks/tfidf.html


# In[1063]:

#This transforms to Tfidf with the original count matrix. This is needed bc the count matrix was adjusted to remove low freq terms.
t0 = time.time()
Tfidf = TfidfTransformer()
convert_Tfidf = Tfidf.fit_transform(doc_term_GT2)
t1 = time.time()
print t1-t0


# In[1064]:

convert_Tfidf.shape


# ### Mutual Information

# In[425]:

def mutualInformation(TD,target):
    '''Calculates the mutual information weights for a term document matrix given a target classifier. Returns the TD matrix with the
    weights applied'''
   
    num_doc = TD.shape[1] #Number of documents
    row_count = (TD != 0).sum(1) #count the number of documents for each term
    
    P_t = row_count/float(num_doc)   
    P_p = (target == 1).sum()/float(num_doc)   
    P_n = 1-P_p
   
    
    index_p = np.where( target == 1 )[0]
    P_pt = (TD[:,index_p] != 0).sum(1)/float(num_doc)
   
    index_n = np.where( target == 0 )[0]
    P_nt = (TD[:,index_n] != 0).sum(1)/float(num_doc)
   
    p = np.log10(P_pt/(P_t*P_p))
    n = np.log10(P_nt/(P_t*P_n))
   
    MI = np.maximum(p,n)
    MI = np.resize(MI,(MI.shape[0],1))
   
    return np.multiply(TD,MI)


# In[1065]:

#Convert doc-term matrix to term-doc
TD=np.array(doc_term_GT2.todense()).T


# In[1066]:

#term-doc matrix weighted by mutual information
TD_MI = mutualInformation(TD,pos_neg)


# In[1067]:

TD_MI.shape


# In[1068]:

#Convert back to Document-term matrix
convert_MI = TD_MI.T
print convert_MI.shape


# ### Word2Vect

# In[ ]:

#http://rare-technologies.com/word2vec-tutorial/
import gensim, logging
import cython


# In[ ]:

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# In[ ]:

#Clean-up reviews for Word2Vect algorithm
pos = ('NN','NNS', #nouns
       'JJ','JJR', 'JJS', #adjectives
       'RB','RBR','RBS', #adverbs
       'VB','VBD','VBG','VBN','VBP','VBZ') #verbs

keep_words = ('but','while')

pos_tagged_w2v = pos_tag(item_toke,pos,keep_words,False) #POS tag
item_lst_w2v=mispelled_remove(pos_tagged_w2v) #Mispelled words removed
stemmed_w2v = SnowballStem(item_lst_w2v) #stemming
dup_removed_w2v = remove_dup(stemmed_w2v) #remove consecutive duplicate words


# In[ ]:

#Build bigram Word2vect model varying the size, s, by 100, 200, and 300
t0 = time.time()

s = 300

bigram_transformer = gensim.models.Phrases(dup_removed_w2v)
model_bi = gensim.models.Word2Vec(bigram_transformer[dup_removed_w2v], size=s, window=5, min_count=10, workers=4)

t1 = time.time()
print t1-t0


# In[ ]:

#terms from model
terms_bi = [word for word in model_bi.vocab.keys()]

#convert terms and vectors to numpy array
vect_lst_bi = [model_bi[word] for word in terms_bi]
w2v_bi_np = np.asarray(vect_lst_bi)

print shape(w2v_bi_np)


# ###### Affinity Propagation

# In[ ]:

from sklearn.cluster import AffinityPropagation


# In[ ]:

#http://scikit-learn.org/stable/auto_examples/cluster/plot_affinity_propagation.html#example-cluster-plot-affinity-propagation-py
# Compute Affinity Propagation
t0 = time.time()

af = AffinityPropagation().fit(w2v_bi_np)
cluster_centers_indices = af.cluster_centers_indices_
cluster_centers = af.cluster_centers_
labels = af.labels_

n_clusters_ = len(cluster_centers_indices)

t1 = time.time()
print t1-t0


# In[ ]:

n_clusters_


# In[ ]:

from sklearn.metrics.pairwise import euclidean_distances


# In[ ]:

#Remove clusters with less than 10 words. Then calculate the ratio of the mean distance from the center and standard deviation for the
#remaining clusters

center_ct = 0
pd_terms = pd.DataFrame(terms_bi) #convert to pandas dataframe for ease of indexing
terms_top10 = []
ratio_rank = []

for c in xrange(n_clusters_):
    terms_cls = pd_terms[labels==c] #terms in each cluster
    
    #if less than 10 => ignore
    if len(terms_cls)>=10:
        dist = euclidean_distances(w2v_bi_np[cluster_centers_indices[c]], w2v_bi_np[labels==c]).T 
        ratio_rank.append(np.mean(dist)/np.std(dist))
        d_sort = np.argsort(dist, axis=None)
        term_list = []
        for i in xrange(1,11):
            term_list.append(terms_cls.values.tolist()[d_sort[-i]])
    
        terms_top10.append(term_list)
    
    center_ct+=1
ratio_rank = np.array(ratio_rank)


# In[ ]:

#extract the top ten clusters based on the ratio
top10_cls = [terms_top10[i] for i in list(np.argsort(ratio_rank,axis=None)) if i<10]


# In[ ]:

#create cluster column names
cls_columns = []
for i in xrange(10):
    cls_columns.append('Cluster'+str(i))


# In[ ]:

#Create csv with top 10 clusters and 10 terms closest to the centroid
path = 'C:\\Users\\jmn251pitt\\Documents\\DePaul\\CSC 594\Project_Final\\w2c_cls_Aff_'+item+'_'+str(s)
path+='.csv'

tmp = open(path,'wb')
forOut=csv.writer(tmp)
forOut.writerow(cls_columns)

for i in xrange(10):
    t = [str(top10_cls[j][i][0]) for j in xrange(len(cls_columns))]
    forOut.writerow(t)
tmp.close()


# ###### Fuzzy Clustering

# In[1127]:

#http://pythonhosted.org/scikit-fuzzy/
#Fuzzy clustering
import skfuzzy as fuzz


# In[ ]:

t0 = time.time()

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(w2v_bi_np.T, 10, 2, error=0.00001, maxiter=1000, init=None) #CHANGE

t1 = time.time()
print t1-t0


# In[ ]:

shape(u)


# In[ ]:

top10_terms = []
for i in xrange(len(cls_columns)):
    term_list=[]
    d_sort = np.argsort(u[i],axis=None)
    for i in xrange(1,11):
            term_list.append(terms_bi[d_sort[-i]])
    top10_terms.append(term_list)


# In[ ]:

shape(top10_terms)


# In[ ]:

#Create csv with top 10 terms by weight for each cluster
path = 'C:\\Users\\jmn251pitt\\Documents\\DePaul\\CSC 594\Project_Final\\w2c_cls_Fuzz_'+item+'_'+str(s)
path+='.csv'

tmp = open(path,'wb')
forOut=csv.writer(tmp)
forOut.writerow(cls_columns)

for i in xrange(10):
    t = [str(top10_terms[j][i]) for j in xrange(len(cls_columns))]
    forOut.writerow(t)
tmp.close()

