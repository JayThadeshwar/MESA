import nltk
from summarizer import Summarizer
import pprint
import itertools
import re
import pke
import string
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from flashtext import KeywordProcessor
import requests
import json
import re
import random
from pywsd.similarity import max_similarity
from pywsd.lesk import adapted_lesk
from pywsd.lesk import simple_lesk
from pywsd.lesk import cosine_lesk
from nltk.corpus import wordnet as wn
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('popular')

# class MCQ:
#     mcq=[]
#     dict1={}
#     listofoptions=[]

#     def __init__(self,m,c):
#         self.mcq=m
#         self.choices=c

def extractMCQ(content):


# content=''' Newton's first law of motion implies that things cannot start, stop, or change direction all by themselves, and it requires some force from the outside to cause such a change. This property of massive bodies to resist changes in their state of motion is called inertia. The first law of motion is also known as the law of inertia.Newton's 1st law states that a body at rest or uniform motion will continue to be at rest or uniform motion until and unless a net external force acts on it.The crucial point here is that if there is no net force resulting from unbalanced forces acting on an object, the object will maintain a constant velocity. If that velocity is zero, then the object remains at rest. And if an additional external force is applied, the velocity will change because of the force.Newton's second law of motion describes what happens to the massive body when acted upon by an external force. The second law of motion states that the force acting on the body is equal to the product of its mass and acceleration.Newton's 2nd law states that the acceleration of an object as produced by a net force is directly proportional to the magnitude of the net force, in the same direction as the net force, and inversely proportional to the object's mass.Newton's second law describes precisely how much an object will accelerate for a given net force.Where f is force, P is linear momentum, m is mass of the object, u is initial velocity, v is final velocity and a is acceleration.In the equation, k is the constant of proportionality, and it is equal to 1 when the values are taken in the SI unit. Hence, the final expression will be.Newton's third law of motion describes what happens to the body when it exerts a force on another body.Newton's 3rd law states that there is an equal and opposite reaction for every action.When two bodies interact, they apply forces on each other that are equal in magnitude and opposite in direction. To understand Newton's third law with the help of an example, let us consider a book resting on a table. The book applies a downward force equal to its weight on the table. According to the third law of motion, the table applies an equal and opposite force to the book. This force occurs because the book slightly deforms the table; as a result, the table pushes back on the book like a coiled spring. Newton;s third law of motion implies the conservation of momentum.'''
    model = Summarizer()
    result = model(content, min_length=60, max_length = 500 , ratio = 0.4)
    summarized_text = ''.join(result)
    #print (summarized_text)


    def get_nouns_multipartite(text):
        out=[]

        extractor = pke.unsupervised.MultipartiteRank()
        extractor.load_document(input=text)
        #    not contain punctuation marks or stopwords as candidates.
        pos = {'PROPN'}
        #pos = {'VERB', 'ADJ', 'NOUN'}
        stoplist = list(string.punctuation)
        stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        stoplist += stopwords.words('english')
        extractor.candidate_selection(pos=pos)
        # 4. build the Multipartite graph and rank candidates using random walk,
        #    alpha controls the weight adjustment mechanism, see TopicRank for
        #    threshold/method parameters.
        extractor.candidate_weighting(alpha=1.1,
                                    threshold=0.75,
                                    method='average')
        keyphrases = extractor.get_n_best(n=20)

        for key in keyphrases:
            out.append(key[0])

        return out

    keywords = get_nouns_multipartite(content) 
    #print (keywords)
    filtered_keys=[]
    for keyword in keywords:
        if keyword.lower() in summarized_text.lower():
            filtered_keys.append(keyword)
            
    print (filtered_keys)

    def tokenize_sentences(text):
        sentences = [sent_tokenize(text)]
        sentences = [y for x in sentences for y in x]
        # Remove any short sentences less than 20 letters.
        sentences = [sentence.strip() for sentence in sentences if len(sentence) > 20]
        return sentences

    def get_sentences_for_keyword(keywords, sentences):
        keyword_processor = KeywordProcessor()
        keyword_sentences = {}
        for word in keywords:
            keyword_sentences[word] = []
            keyword_processor.add_keyword(word)
        for sentence in sentences:
            keywords_found = keyword_processor.extract_keywords(sentence)
            for key in keywords_found:
                keyword_sentences[key].append(sentence)

        for key in keyword_sentences.keys():
            values = keyword_sentences[key]
            values = sorted(values, key=len, reverse=True)
            keyword_sentences[key] = values
        return keyword_sentences

    sentences = tokenize_sentences(summarized_text)
    keyword_sentence_mapping = get_sentences_for_keyword(filtered_keys, sentences)
            
    print (keyword_sentence_mapping)
    print("\nENd")

    # Distractors from Wordnet
    def get_distractors_wordnet(syn,word):
        distractors=[]
        word= word.lower()
        orig_word = word
        if len(word.split())>0:
            word = word.replace(" ","_")
        hypernym = syn.hypernyms()
        if len(hypernym) == 0: 
            return distractors
        for item in hypernym[0].hyponyms():
            name = item.lemmas()[0].name()
            #print ("name ",name, " word",orig_word)
            if name == orig_word:
                continue
            name = name.replace("_"," ")
            name = " ".join(w.capitalize() for w in name.split())
            if name is not None and name not in distractors:
                distractors.append(name)
        return distractors

    def get_wordsense(sent,word):
        word= word.lower()
        
        if len(word.split())>0:
            word = word.replace(" ","_")
        
        
        synsets = wn.synsets(word,'n')
        if synsets:
            wup = max_similarity(sent, word, 'wup', pos='n')
            adapted_lesk_output =  adapted_lesk(sent, word, pos='n')
            lowest_index = min (synsets.index(wup),synsets.index(adapted_lesk_output))
            return synsets[lowest_index]
        else:
            return None

    # Distractors from conceptnet
    def get_distractors_conceptnet(word):
        word = word.lower()
        original_word= word
        if (len(word.split())>0):
            word = word.replace(" ","_")
        distractor_list = [] 
        url = "http://api.conceptnet.io/query?node=/c/en/%s/n&rel=/r/PartOf&start=/c/en/%s&limit=5"%(word,word)
        obj = requests.get(url).json()

        for edge in obj['edges']:
            link = edge['end']['term'] 

            url2 = "http://api.conceptnet.io/query?node=%s&rel=/r/PartOf&end=%s&limit=10"%(link,link)
            obj2 = requests.get(url2).json()
            for edge in obj2['edges']:
                word2 = edge['start']['label']
                if word2 not in distractor_list and original_word.lower() not in word2.lower():
                    distractor_list.append(word2)
                    
        return distractor_list

    key_distractor_list = {}

    for keyword in keyword_sentence_mapping:
        print("Above wordsense")
        wordsense = get_wordsense(keyword_sentence_mapping[keyword][0],keyword)
        print(wordsense)
        if wordsense:
            distractors = get_distractors_wordnet(wordsense,keyword)
            if len(distractors) ==0:
                distractors = get_distractors_conceptnet(keyword)
            if len(distractors) != 0:
                key_distractor_list[keyword] = distractors
        else:
            
            distractors = get_distractors_conceptnet(keyword)
            if len(distractors) != 0:
                key_distractor_list[keyword] = distractors

    mcq=[]
    listofoptions=[]
    finaldict={}
    index = 1
    for each in key_distractor_list:
        sentence = keyword_sentence_mapping[each][0]
        pattern = re.compile(each, re.IGNORECASE)
        output = pattern.sub( " _______ ", sentence)
        print ("%s)"%(index),output)
        mcq.append(output)
        choices = [each.capitalize()] + key_distractor_list[each]
        top4choices = choices[:4]
        listofoptions.append(top4choices)
        #random.shuffle(top4choices)
        optionchoices = ['a','b','c','d']
        for idx,choice in enumerate(top4choices):
            print ("\t",optionchoices[idx],")"," ",choice)
        print ("\nMore options: ", choices[4:20],"\n\n")
        index = index + 1
    print(mcq)
    print(listofoptions)
    finaldict = {mcq[i]: listofoptions[i] for i in range(len(mcq))}
    print("Final ans=",finaldict)
    return finaldict
