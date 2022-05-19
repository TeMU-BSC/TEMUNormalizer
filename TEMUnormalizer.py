#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 16:53:49 2020

@author: crodri

Baseline Term normalizer through exact and fuzzy match.

(Will add embedding strategy later)



"""

import csv, os, sys, time, pickle
#from fuzzywuzzy import process
from rapidfuzz import process#, utils
#from collections import defaultdict
#from gensim import corpora
#import os
#import nltk
from nltk.tokenize import word_tokenize
#For TEST
#os.chdir("/home/crodri/GIT/TEMUNorm/")
#filepath = "./tsv_dictionaries/SpanishSnomed.tsv"
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def loadDict(filepath):
    import re
    r = re.compile(r'\(.*\)$')
    reference_dict = {}
    for duplo in csv.reader(open(filepath),dialect='excel',delimiter="\t"):
        term = duplo[0]
        code = duplo[1]
        codes = code.split("|")
        term = r.sub('',term).rstrip().lower()
        if term in reference_dict.keys():
            previous_codes = reference_dict[term]
            newlist = list(set(codes).union(set(previous_codes)))
            reference_dict[term] = newlist
        else: 
            reference_dict[term] = codes
    print("Loaded dictionary from: ",filepath)
    print(len(reference_dict)," entries")
    return reference_dict


def loadTermList(apath):
    termdic = {}
    for term in open(apath,'r').readlines():
        termdic[term.lower().rstrip()] = ''
    return termdic

def loadAnn(apath,entidades=None):
    """
    Parameters
    ----------
    apath : TYPE
        DESCRIPTION.
    entidades : TYPE, optional
        DESCRIPTION. The default is [].

    Returns
    -------
    termlist : TYPE
        DESCRIPTION.
        
    """
    termdic = {}
    for line in csv.reader(open(apath),dialect='excel',delimiter="\t"):
        if entidades:
            if line[1].split(" ")[0] in entidades:
                termdic[line[-1].lower()] = ''
        else:
            termdic[line[-1].lower()] = ''
    return termdic

def notEmpty(termdic):
    termlist = []
    for t in termdic:
        if termdic[t] == '':
            termlist.append(t)
    return termlist

def directMatch(termdic,reference_dict):
    for term in termdic:
        try:
            snomedid = reference_dict[term]
            #print("ExactMatch: ",snomedid)
            termdic[term] = [[snomedid,100.0]]
            #termlist.remove(term)
        except KeyError:
            pass
            #print("For term: ",str2Match)
    return termdic

def fuzzyMatch(termdic,reference_dict,umbral):
    """
    umbral entre 0 y 100
    """
    #import time
    t1 = time.time()
    allterms = reference_dict.keys()
    testerms = notEmpty(termdic)
    print("Will search: ",len(testerms)," using fuzzy match")
    for term in testerms:
        highest = process.extractOne(term,allterms,processor=None, score_cutoff=93)
        candidatelist = []
        if highest:
            if highest[-1] >= umbral:
                termdic[term] = [[reference_dict[highest[0]],highest[-1]]]
            else:
                pass
    t2 = time.time()
    print("Fuzzy matching applied in ",(t2-t1)/60," minutes")
    return termdic

def alt_cosine(x,y):
    return np.inner(x,y)/np.sqrt(np.dot(x,x)*np.dot(y,y)) #~25x faster than sklearn

def sentenceTransformerMatch(termdic,reference_dict,reference_dict_vec='',umbral=93):
    """
    umbral entre 0 y 100
    """
    #import time
    t1 = time.time()
    # If the vectorisation of the reference dict is already done, import it
    # else, compute it
    model = SentenceTransformer('hiiamsid/sentence_similarity_spanish_es')
    allterms = list(reference_dict.keys())
    if os.path.isfile(reference_dict_vec):
        allterms_vec = np.load(reference_dict_vec)
    else:
        allterms_vec = model.encode(allterms)
    testerms = notEmpty(termdic)
    print("Will search: ",len(testerms)," using sentence transformer match")
    testerms_vec = model.encode(testerms)
    for i in range(len(testerms)):
        cosine = []
        for j in range(len(allterms)):
            #cosine.append(cosine_similarity(testerms_vec[i,:].reshape(1,-1), allterms_vec[j,:].reshape(1,-1))[0][0])
            cosine.append(alt_cosine(np.squeeze(testerms_vec[i,:].reshape(1,-1)), 
                                            np.squeeze(allterms_vec[j,:].reshape(1,-1))))
        highest = [allterms[cosine.index(max(cosine))]]
        highest.append(max(cosine))
        if highest:
            if highest[-1]*100 >= umbral: # Cosine similarity is between 0 and 1
                termdic[testerms[i]] = [[reference_dict[highest[0]],highest[-1]]]
            else:
                pass
    t2 = time.time()
    print("Sentence transformer matching applied in ",(t2-t1)/60," minutes")
    return termdic


#-To Add Later <---------------------
def createW2VecIndex(reference_dict):
    from gensim.corpora import Dictionary
    from gensim.models import Word2Vec, WordEmbeddingSimilarityIndex
    from gensim.similarities import SoftCosineSimilarity, SparseTermSimilarityMatrix
    print("Prepare Word2Vec model")
    import time
    t1 = time.time()
    corpus = []
    #reference = []
    for term in reference_dict:
        corpus.append(word_tokenize(term))
        #reference.append(term)
    model = Word2Vec(corpus, size=20, min_count=1)  # train word-vectors
    termsim_index = WordEmbeddingSimilarityIndex(model.wv)#<----
    dictionary = Dictionary(corpus)
    bow_corpus = [dictionary.doc2bow(document) for document in corpus]
    similarity_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary)  # construct similarity matrix
    docsim_index = SoftCosineSimilarity(bow_corpus, similarity_matrix, num_best=3)   
    t2 = time.time()
    print(" W2v index and dictionary in ",(t2-t1)/60," minutes")
    import pickle
    f = open("./models/W2VecIndexes.bin",'wb')
    pickle.dump((docsim_index,dictionary),f)
    return   docsim_index,dictionary        
#-To Add Later <---------------------
def W2VSimilarity(termdic,reference_dict,umbral, create=None):
    sentences = [x for x in reference_dict]
    # from gensim.corpora import Dictionary
    # from gensim.models import Word2Vec, WordEmbeddingSimilarityIndex
    # from gensim.similarities import SoftCosineSimilarity, SparseTermSimilarityMatrix
    if create:
        docsim_index,dictionary = createW2VecIndex(reference_dict)
    else:
        #import pickle
        duplo = pickle.load(open("./models/W2VecIndexes.bin",'rb'))
        docsim_index,dictionary = duplo[0],duplo[-1]
    import time
    t1 = time.time()
    testerms = notEmpty(termdic)
    for term in testerms:
        #hecho = termdic[term]
        query = word_tokenize(term)
        sims = docsim_index[dictionary.doc2bow(query)]
        results = []
        for each in sims:
            if each[-1] >= umbral:
                sentence = sentences[each[0]]
                #t = " ".join(sentence)
                results.append([reference_dict[sentence],(each[-1]*100)])
                if results:
                    termdic[term] = results
    t2 = time.time()
    print(" W2v processing in ",(t2-t1)/60," minutes")
    return termdic

#fileout = "test_out.tsv"
def writeOut(termdic,fileout):
    w = open(fileout,"w")
    wr = csv.writer(w,dialect='excel',delimiter="\t")
    for term in termdic:
        #print(term)
        objeto = termdic[term]
        if objeto == '':
            wr.writerow([term,'',0.0])
        else:
            code,score = objeto[0]
            wr.writerow([term,', '.join(code),score])
    w.close()

from optparse import OptionParser
def main(argv=None):
    parser = OptionParser()
    parser.add_option("-d", "--dictionary", dest="reference_dict",
                    help="tab-separated (term to code) file to create reference dictionary from",default="SpanishSnomed.tsv")
    parser.add_option("-f", "--fileout", dest="fileout", help="output file, tab-separated values extension (.tsv)",default="termlist_normalized.tsv")
  
    parser.add_option("-t", "--terms", dest="termlist", help="file with term list to normalize, one per line")
    parser.add_option("-a", "--ann", dest="brat", help=" treat input file as brat .ann file with term list to normalize, one per line", default=None)
    parser.add_option("-e", "--entities", dest="entities", help="give a list (comma-separated) of names of entities to normalize. Otherwise, will try to normalize everything\t e.g.: python TEMUnormalizer.py -d ./tsv_dictionaries/SpanishSnomed.tsv -f normalized_list_snomed_from_ann.tsv -t S0004-06142005000200004-1.ann -a 1 -e ENFERMEDAD,FARMACOS,FARMACOS-2", default=None)
    parser.add_option("-u", "--umbral", dest="umbral", help=" threshold for fuzzy search (default 93)",type="int", default=93)
    #parser.add_option("-r", "--redact", dest="redact", help="do not write target sentences", default=None)
    (options, args) = parser.parse_args(argv)
    
    print("Load reference dictionary from", options.reference_dict)
    reference_dict = loadDict(options.reference_dict)
    if options.termlist:
        print("load term list: ",options.termlist)
        if options.brat:
            if options.entities:
                entitieslist = options.entities.split(',') 
                termdic = loadAnn(options.termlist,entitieslist)
            else:   
                termdic = loadAnn(options.termlist)
        else:
            termdic = loadTermList(options.termlist)
        
        initiatewith = len(notEmpty(termdic))
        print("number of terms to test: ", initiatewith)
        
        t1 = time.time()
        print("First  try exact Match")
        termdic = directMatch(termdic,reference_dict)
        print("number of terms missing after direct match: ", len(notEmpty(termdic)))
        
        print("fuzzy match")
        termdic = fuzzyMatch(termdic,reference_dict,options.umbral)
        endedwith = len(notEmpty(termdic))
        print("number of terms missing after fuzzy match: ", endedwith)
        
        print("Sentence transformer match")
        termdic = sentenceTransformerMatch(termdic,reference_dict,umbral=options.umbral)
        endedwith = len(notEmpty(termdic))
        print("number of terms missing after Sentence transformer match: ", endedwith)
        
        percent = (endedwith*100)/initiatewith
        print(percent," % NOT found")
        writeOut(termdic,options.fileout)
        t2 = time.time()
        print(" Overall processing in ",(t2-t1)/60," minutes")
    else:
        print("Specify a term list file to process")
if __name__ == "__main__":
  sys.exit(main())
