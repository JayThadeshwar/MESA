import re
import nltk
import spacy

import numpy as np
from nltk import sent_tokenize, word_tokenize
from nltk.cluster.util import cosine_distance

MULTIPLE_WHITESPACE_PATTERN = re.compile(r"\s+", re.UNICODE)

def normalize_whitespace(text):
    """
    Translates multiple whitespace into single space character.
    If there is at least one new line character chunk is replaced
    by single LF (Unix new line) character.
    """
    return MULTIPLE_WHITESPACE_PATTERN.sub(_replace_whitespace, text)


def _replace_whitespace(match):
    text = match.group()

    if "\n" in text or "\r" in text:
        return "\n"
    else:
        return " "


def is_blank(string):
    """
    Returns `True` if string contains only white-space characters
    or is empty. Otherwise `False` is returned.
    """
    return not string or string.isspace()


def get_symmetric_matrix(matrix):
    """
    Get Symmetric matrix
    :param matrix:
    :return: matrix
    """
    return matrix + matrix.T - np.diag(matrix.diagonal())


def core_cosine_similarity(vector1, vector2):
    """
    measure cosine similarity between two vectors
    :param vector1:
    :param vector2:
    :return: 0 < cosine similarity value < 1
    """
    return 1 - cosine_distance(vector1, vector2)


class TextRank4Sentences():
    def __init__(self):
        self.damping = 0.85  # damping coefficient, usually is .85
        self.min_diff = 1e-5  # convergence threshold
        self.steps = 100  # iteration steps
        self.text_str = None
        self.sentences = None
        self.pr_vector = None

    def _sentence_similarity(self, sent1, sent2, stopwords=None):
        if stopwords is None:
            stopwords = []

        sent1 = [w.lower() for w in sent1]
        sent2 = [w.lower() for w in sent2]

        all_words = list(set(sent1 + sent2))

        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)

        # build the vector for the first sentence
        for w in sent1:
            if w in stopwords:
                continue
            vector1[all_words.index(w)] += 1

        # build the vector for the second sentence
        for w in sent2:
            if w in stopwords:
                continue
            vector2[all_words.index(w)] += 1

        return core_cosine_similarity(vector1, vector2)

    def _build_similarity_matrix(self, sentences, stopwords=None):
        # create an empty similarity matrix
        sm = np.zeros([len(sentences), len(sentences)])

        for idx1 in range(len(sentences)):
            for idx2 in range(len(sentences)):
                if idx1 == idx2:
                    continue

                sm[idx1][idx2] = self._sentence_similarity(sentences[idx1], sentences[idx2], stopwords=stopwords)

        # Get Symmeric matrix
        sm = get_symmetric_matrix(sm)

        # Normalize matrix by column
        norm = np.sum(sm, axis=0)
        sm_norm = np.divide(sm, norm, where=norm != 0)  # this is ignore the 0 element in norm

        return sm_norm

    def _run_page_rank(self, similarity_matrix):

        pr_vector = np.array([1] * len(similarity_matrix))

        # Iteration
        previous_pr = 0
        for epoch in range(self.steps):
            pr_vector = (1 - self.damping) + self.damping * np.matmul(similarity_matrix, pr_vector)
            if abs(previous_pr - sum(pr_vector)) < self.min_diff:
                break
            else:
                previous_pr = sum(pr_vector)

        return pr_vector

    def _get_sentence(self, index):

        try:
            return self.sentences[index]
        except IndexError:
            return ""

    def get_top_sentences(self, number=5):

        top_sentences = []

        if self.pr_vector is not None:

            sorted_pr = np.argsort(self.pr_vector)
            sorted_pr = list(sorted_pr)
            sorted_pr.reverse()

            index = 0
            for epoch in range(number):
                sent = self.sentences[sorted_pr[index]]
                sent = normalize_whitespace(sent)
                top_sentences.append(sent)
                index += 1

        return top_sentences

    def analyze(self, text, stop_words=None):
        self.text_str = text
        self.sentences = sent_tokenize(self.text_str)

        tokenized_sentences = [word_tokenize(sent) for sent in self.sentences]

        similarity_matrix = self._build_similarity_matrix(tokenized_sentences, stop_words)

        self.pr_vector = self._run_page_rank(similarity_matrix)

# -------------------------------------------------------------------------
nlp = spacy.load('en_core_web_sm')

def cleanData(text):    
    text = re.sub('\n',' ',str(text))
    text = re.sub("'s",'',str(text))
    text = re.sub("-",' ',str(text))
    
    result = ''
    for word in text.split(' '):
      if(not re.search("[^a-zA-Z/./s]",word)):
        result += " " + word
    return result

def getSentences(data):
    text = re.split('[.?]', data)
    clean_sent = []
    for sent in text:
        cln_s = sent.strip()
        if(cln_s != ""):
            clean_sent.append(cln_s)
    return clean_sent        

def rule2_mod(text,index):    
    doc = nlp(text)
    phrase = ''
    
    for token in doc:        
        if token.i == index:            
            for subtoken in token.children:
                if (subtoken.pos_ == 'ADJ'):
                    phrase += ' '+subtoken.text
            break
    
    return phrase

# rule 1 modified function
def rule1_mod(text):
    
    doc = nlp(text)
    
    sent = dict()
    sent['sentence'] = text
    
    for token in doc:
        # root word
        if (token.pos_=='VERB'):
            
            phrase =''
            
            # only extract noun or pronoun subjects
            for sub_tok in token.lefts:
                
                if (sub_tok.dep_ in ['nsubj','nsubjpass']) and (sub_tok.pos_ in ['NOUN','PROPN','PRON']):
                    
                    # look for subject modifier
                    adj = rule2_mod(text,sub_tok.i)
                    
                    phrase += adj + ' ' + sub_tok.text
                    sent['subject'] = phrase.strip()

                    # save the root word of the word
                    phrase += ' '+token.lemma_ 
                    sent['verb'] = token.lemma_

                    # check for noun or pronoun direct objects
                    for sub_tok in token.rights:
                        
                        if (sub_tok.dep_ in ['dobj']) and (sub_tok.pos_ in ['NOUN','PROPN']):
                            
                            # look for object modifier
                            adj = rule2_mod(text,sub_tok.i)
                            sent['object'] = (adj+' '+sub_tok.text).strip()

                            phrase += adj+' '+sub_tok.text
                            sent['phrase'] = phrase.strip()    
            
    return sent

# -------------------------------------------------------------------------

def generateGrammarDetails(content, k):    
    tr4sh = TextRank4Sentences()
    tr4sh.analyze(content)
    top_k_sent = tr4sh.get_top_sentences(k)

    response = []
    nal_pos = ['DET','PUNCT','SPACE', 'AUX', 'NUM']

    for sent in top_k_sent:        
        posDict = {}
        posDict['sentence'] = sent
        sent = nlp(sent)  
        for tok in sent:    
            if(tok.pos_ not in nal_pos): 
                expl = spacy.explain(tok.pos_)
                if(tok.pos_ not in posDict):
                    posDict[expl] = list()     
                res = {}
                res['word'] = tok.text
                res['tagExp'] = spacy.explain(tok.tag_)
                posDict[expl].append(res)
        response.append(posDict)
    
    svo = []

    preProcessedData = cleanData(content)
    sentences = getSentences(preProcessedData)

    for sent in sentences:
        res = rule1_mod(sent)
        if('subject' in res and 'verb' in res and 'object' in res):            
            svo.append(res)

    finalResp = {"grammarticalInfo": response, "svoInfo": svo}    
    return finalResp

# -------------------------------------------------------------------------
