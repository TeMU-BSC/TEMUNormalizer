#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 15:17:27 2020

@author: crodri
"""

import csv
snomedict = {}
snomedfull = csv.reader(open("/home/crodri/BSC/mappings/SNOMED/SnomedCT_Spanish_Edition/SnomedCT_SpanishRelease-es_PRODUCTION_20200430T120000Z/RF2Release/Snapshot/Terminology/sct2_Description_SpanishExtensionSnapshot-es_INT_20200430.txt"),dialect='excel',delimiter="\t")

for x in snomedfull:
    if x[0] == 'id':
        pass
    elif x[2] == '1':
        try:
            if x[7] in snomedict.keys():
                keylist = snomedict[x[7].lower()]
                if x[4] in keylist:
                    pass
                else:
                    keylist.append(x[4])
                    snomedict[x[7].lower()] = keylist
            else:
                snomedict[x[7].lower()] = [x[4]]
        except IndexError:
            pass

#import pickle
salida = csv.writer(open("/home/crodri/GIT/TEMUNorm/tsv_dictionaries/SpanishSnomed.tsv",'w'),dialect='excel',delimiter="\t")
n = 0
m = 0
for r in snomedict:
    cs = snomedict[r]
    codes = "|".join(cs)
    salida.writerow([r,codes])
    n += 1
print(n)

icd10 = {}
reverseicd = {}
f = 0
r = 0
for x in csv.reader(open("/home/crodri/BSC/mappings/SNOMED/SNOMED_CT_to_ICD-10-CM_Resources_20200301/SNOMED_CT_to_ICD-10-CM_Resources_20200301/tls_Icd10cmHumanReadableMap_US1000124_20200301.tsv"),dialect='excel',delimiter="\t"):
    if x[0] == 'id':
        pass
    elif x[2] == '1':
        if x[5] in icd10.keys():
            #print("already found: ",x[5],"\t", x[11])
            f += 1
        else:
            if x[11] == '':
                print( x)
            else:
                icd10[x[5]] = x[11]
        if x[11] == '':
            pass
        else:
            if x[11]in reverseicd.keys():
                print("Repeated ",x[11])
                r += 1
            else:
                if x[5] == '':
                    print(x)
                else:
                    reverseicd[x[11]] = x[5]


reverSnomed = {}
for t in snomedict:
    codes = snomedict[t]
    for c in codes:
        if c in reverSnomed:
            listado = reverSnomed[c]
            listado.append(t)
            reverSnomed[c] = listado
        else:
            reverSnomed[c] = [t]
len(reverSnomed)

mappICD2SnoTerms = {}

non = 0
for s in icd10:
    i = icd10[s]
    try:
        termos = reverSnomed[s]
        mappICD2SnoTerms[i] = [s,termos]
    except KeyError:
        non += 1
print("Keys not found: ",non)
 
salidacie = csv.writer(open("/home/crodri/GIT/TEMUNorm/tsv_dictionaries/SpanishCIE10.tsv",'w'),dialect='excel',delimiter="\t")

for i in mappICD2SnoTerms:
    snomed,listaterminos = mappICD2SnoTerms[i]
    for t in listaterminos:
        salidacie.writerow([t,i,snomed])
