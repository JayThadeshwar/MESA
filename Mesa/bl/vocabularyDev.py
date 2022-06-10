import nltk

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

import string
import pandas as pd
import numpy as np
import spacy
import requests
import textstat

from nltk import tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from rake_nltk import Rake
from collections import OrderedDict
from spacy.lang.en.stop_words import STOP_WORDS

#TextRank ALgo
nlp = spacy.load('en_core_web_sm')

class TextRank4Keyword():
    """Extract keywords from text"""
    
    def __init__(self):
        self.d = 0.85 # damping coefficient, usually is .85
        self.min_diff = 1e-5 # convergence threshold
        self.steps = 10 # iteration steps
        self.node_weight = None # save keywords and its weight

    
    def set_stopwords(self, stopwords):  
        """Set stop words"""
        for word in STOP_WORDS.union(set(stopwords)):
            lexeme = nlp.vocab[word]
            lexeme.is_stop = True
    
    def sentence_segment(self, doc, candidate_pos, lower):
        """Store those words only in cadidate_pos"""
        sentences = []
        for sent in doc.sents:
            selected_words = []
            for token in sent:
                # Store words only with cadidate POS tag
                if token.pos_ in candidate_pos and token.is_stop is False:
                    if lower is True:
                        selected_words.append(token.text.lower())
                    else:
                        selected_words.append(token.text)
            sentences.append(selected_words)
        return sentences
        
    def get_vocab(self, sentences):
        """Get all tokens"""
        vocab = OrderedDict()
        i = 0
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = i
                    i += 1
        return vocab
    
    def get_token_pairs(self, window_size, sentences):
        """Build token_pairs from windows in sentences"""
        token_pairs = list()
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(i+1, i+window_size):
                    if j >= len(sentence):
                        break
                    pair = (word, sentence[j])
                    if pair not in token_pairs:
                        token_pairs.append(pair)
        return token_pairs
        
    def symmetrize(self, a):
        return a + a.T - np.diag(a.diagonal())
    
    def get_matrix(self, vocab, token_pairs):
        """Get normalized matrix"""
        # Build matrix
        vocab_size = len(vocab)
        g = np.zeros((vocab_size, vocab_size), dtype='float')
        for word1, word2 in token_pairs:
            i, j = vocab[word1], vocab[word2]
            g[i][j] = 1
        
        # Get Symmeric matrix
        g = self.symmetrize(g)
        
        # Normalize matrix by column
        norm = np.sum(g, axis=0)
        g_norm = np.divide(g, norm, where=norm!=0) # this is ignore the 0 element in norm
        
        return g_norm
    
    def getNodeWeight(self):
      node_weight = OrderedDict(sorted(self.node_weight.items(), key=lambda t: t[1], reverse=True))
      return node_weight
        
    def analyze(self, text, 
                candidate_pos=['NOUN', 'PROPN'], 
                window_size=4, lower=False, stopwords=list()):
        """Main function to analyze text"""
        
        # Set stop words
        self.set_stopwords(stopwords)
        
        # Pare text by spaCy
        doc = nlp(text)
        
        # Filter sentences
        sentences = self.sentence_segment(doc, candidate_pos, lower) # list of list of words
        
        # Build vocabulary
        vocab = self.get_vocab(sentences)

        # Get token_pairs from windows
        token_pairs = self.get_token_pairs(window_size, sentences)

        # Get normalized matrix
        g = self.get_matrix(vocab, token_pairs)
        
        # Initionlization for weight(pagerank value)
        pr = np.array([1] * len(vocab))
        
        # Iteration
        previous_pr = 0
        for epoch in range(self.steps):
            pr = (1-self.d) + self.d * np.dot(g, pr)
            if abs(previous_pr - sum(pr))  < self.min_diff:
                break
            else:
                previous_pr = sum(pr)

        # Get weight for each node
        node_weight = dict()
        for word, index in vocab.items():
            node_weight[word] = pr[index]
        
        self.node_weight = node_weight


english_most_common_10k = 'https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-usa-no-swears.txt'
response = requests.get(english_most_common_10k)
dt = response.text
set_of_common_words = {x for x in dt.split('\n')}
set_of_common_words.remove('regime')
set_of_common_words.remove('exemption')
set_of_common_words.remove('doctrine')
set_of_common_words.remove('manor')

import re

def filterCommonWords(listOfExtractedWords):
    
    #Filter Words        
    li = []
    for dword in listOfExtractedWords:
        if(dword not in set_of_common_words and (not re.search('[^a-zA-Z]', dword))):
            li.append(dword)
    return li 


#Final ALgo
def riOFWord(inputText):
  # TextRank scores
  tr4w = TextRank4Keyword()
  tr4w.analyze(inputText, candidate_pos = ['NOUN','VERB','ADJ'], window_size=2, lower=True)
  node_weights = tr4w.getNodeWeight()

  # Making dict for filtered words of TextRank
  filteredwordsTextrank=filterCommonWords(node_weights.keys())
  #print("Manually extracted difficult words:",filteredwordsTextrank[:lenClsTrue])
  dictOfTR = {}
  for word in filteredwordsTextrank:
    dictOfTR[word] = node_weights[word]

  # MinMax Normalization on TR
  maxx = list(dictOfTR.values())[0]
  minx = list(dictOfTR.values())[len(dictOfTR)-1]
  for word in dictOfTR:
    dictOfTR[word] = (dictOfTR[word] - minx)/(maxx-minx)

  # Initializations
  textstat.set_lang('en')
  candidate_pos = ['NOUN','VERB','ADJ']
  stop_words = set(stopwords.words("english"))
  lemmatizer = WordNetLemmatizer()
#   lenClassTrue = len(classTrue)

  sents = tokenize.sent_tokenize(inputText)
  listOfWords = {}
  for sent in sents:
    # Selecting based on candidate POS
    posTaggedSent = nlp(sent)
    res = [lemmatizer.lemmatize(word.text.lower()) for word in posTaggedSent if word.pos_ in candidate_pos]    
    proSent = ' '.join(res)

    # Removing stopwords and common words
    fin = [word for word in proSent.split(' ') if word.lower() not in set_of_common_words and word.lower() not in stop_words and (not re.search('[^a-zA-Z]', word))]    
    proFin = ' '.join(fin)

    # Scoring words based in readability index
    if(len(proFin) != 0):
      for word in proFin.split(' '):
        if(word not in listOfWords):
          listOfWords[word] = textstat.dale_chall_readability_score(proSent) / len(res)
  
  # Sorting the resulting list
  sortedFinalList = dict(sorted(listOfWords.items(),key= lambda x:x[1],reverse=True))

  # MinMax Normalization on words found by readability index
  resultWords = {}
  maxx = list(sortedFinalList.values())[0]
  minx = list(sortedFinalList.values())[len(sortedFinalList)-1]
  for word in sortedFinalList:
    resultWords[word] = (sortedFinalList[word] - minx)/(maxx-minx)

  # Adding TR and readability index scores
  finalAns = {}
  for word in resultWords:
    if(word in dictOfTR):
      finalAns[word] = dictOfTR[word] + resultWords[word]    

  return finalAns

class WordInformation:
    word=''
    definition=[]
    synonyms=[]
    antonyms=[]
    example=[]
    #audioLink=[]
    # translatedWord=''

    def __init__(self,w,d,s,a,e):
        self.word=w
        self.definition=d
        self.synonyms=s
        self.antonyms=a
        self.example=e
        #self.audioLink=ad
        # self.translatedWord=t    

def findAudio(data):
  if 'phonetics' in data[0]:
    if 'phonetics' != None:
      if 'audio' in data[0]['phonetics'][0] and 'audio' != None:
        ad_link=data[0]['phonetics'][0]['audio']
        return ad_link
    else:
        return None

def findExample(data):
  if 'meanings' in data[0]:
    if 'meanings' == None:
      return
    else:
      if 'definitions' in data[0] ['meanings'][0]:
        if 'definitions' == None:
          return
        else:
          if 'example' in data[0]['meanings'][0]['definitions'][0]:
            ex=data[0]['meanings'][0]["definitions"][0]['example']
            return ex


def extractKeywordsFromContent(content):    
    res = riOFWord(content)
    result = list()

    for word in list(res.keys())[:10]:
        syns = wordnet.synsets(word)
        synonyms = set()
        antonyms = set()
        definition = list()
        example=[]
        #audioLink=[]
        req4 = requests.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}")
        newdata=req4.json()
        #audiovar=None
        #audiovar=findAudio(newdata)
        exvar=None
        exvar=findExample(newdata)
        #audioLink.append(audiovar)
        example.append(exvar)

        for syn in syns:
            synon = [word.name() for word in syn.lemmas()]
            anton= [lemma.antonyms()[0].name() for lemma in syn.lemmas() if lemma.antonyms()]
            synonyms.update(synon)
            antonyms.update(anton)
            definition.append(syn.definition())    

        result.append(WordInformation(word, definition, list(synonyms), list(antonyms),list(example)).__dict__)
    return result
