#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 11:31:02 2020

@author: crodri
"""

import pickle
import os
os.chdir("/home/crodri/GIT/TEMUNorm/")
#pickle.load(open("resultsdic.bin","rb"))


def notEmpty(termdic):
    termlist = []
    for t in termdic:
        if termdic[t] == ('', '0.0'):
            termlist.append(t)
    return termlist

def loadGSDict():
    import re, csv
    #r = re.compile(r'\(.*\)$')
    reference_dict = {}
    for term, code in csv.reader(open("snomed_sp_GS.tsv"),dialect='excel',delimiter="\t"):
        #term = r.sub('',term).rstrip()
        reference_dict[term.lower()] = code
    #print("Loaded dictionary from: ",filepath)
    #print(len(reference_dict)," GS entries")
    return reference_dict

def loadResultsDict(inputdic):
    import re, csv
    #r = re.compile(r'\(.*\)$')
    resultsdict = {}
    for entry in csv.reader(open(inputdic),dialect='excel',delimiter="\t"):
        term, code, score = entry[0],entry[1],entry[-1]
        #term = r.sub('',term).rstrip()
        resultsdict[term] = (code,score)
    print(len(resultsdict)," test results entries")
    return resultsdict
gsdict = loadGSDict()
resultsdic = loadResultsDict("normalized_list_snomed.tsv")
test = []
reference = []
nontermed = notEmpty(resultsdic)
print(len(nontermed),"With no results")
percent = round((len(nontermed)*100)/ len(resultsdic),ndigits=2)
print(percent," % not found: ")
print("On the % ",(100.0-percent),"found: ")
notfounInGS = []
for term in resultsdic:
    if term in nontermed:
        pass
    else:
        try:
            code,score = resultsdic[term]
            snomed = gsdict[term]
            codes = code.split(',')
            if snomed in codes:
                test.append(snomed)
            else:
                test.append(codes[0])
            reference.append(snomed)
        except KeyError:
            notfounInGS.append(term)

from nltk.metrics.scores import *          
print("Accuracy: ",accuracy(reference,test))
print("Precision: ",precision(set(reference),set(test)))
print("Recall: ",recall(set(reference),set(test)))
print("F-measure",f_measure(set(reference), set(test), alpha=0.5))  