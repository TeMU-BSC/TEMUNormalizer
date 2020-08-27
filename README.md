# TEMUnormalizer

## Description
Baseline term normalizer to find Snomed and CIE-10 codes
Two very simple strategies for now:
- Direct Match from dictionary
- Fuzzy match using the rapidfuzz library (https://github.com/maxbachmann/rapidfuzz)

-In code, there is a third strategy using a w2v matrix for soft cosine indexing, but is not operative


- [convertSnomed2tsv.py] Genera los diccionarios de Snomed y CIE-10 desde archivos de la distribución en Espñol de esos recursos

## Usage

python TEMUnormalizer.py -h

Usage: TEMUnormalizer.py [options]

<pre>

Options:

-h, --help            show this help message and exit
  -d REFERENCE_DICT, --dictionary=REFERENCE_DICT
                        tab-separated (term to code) file to create reference
                        dictionary from
                        
  -f FILEOUT, --fileout=FILEOUT
                        output file, tab-separated values extension (.tsv)
                        
  -t TERMLIST, --terms=TERMLIST
                        file with term list to normalize, one per line
                        
  -a BRAT, --ann=BRAT    treat input file as brat .ann file with term list to
                        normalize, one per line
                        
  -e ENTITIES, --entities=ENTITIES
                        give a list (comma-separated) of names of entities to
                        normalize. Otherwise, will try to normalize everything
 
 </pre> 
 ## Example
 
 <pre> 

e.g.: python TEMUnormalizer.py -d./tsv_dictionaries/SpanishSnomed.tsv 
  -f  normalized_list_snomed_from_ann.tsv 
  -t S0004-06142005000200004-1.ann 
 -a 1 
 -e ENFERMEDAD,FARMACOS,FARMACOS-2
 -u UMBRAL, --umbral=UMBRAL threshold for fuzzy search (default 93)
 </pre>
<pre> 
 python TEMUnormalizer.py -t test_term_list.txt -f normalized_list_snomed.tsv
Load reference dictionary from ./tsv_dictionaries/SpanishSnomed.tsv
Loaded dictionary from:  ./tsv_dictionaries/SpanishSnomed.tsv
565920  entries
load term list:  test_term_list.txt
number of terms to test:  295
First  try exact Match
number of terms missing after direct match:  113
fuzzy match
Will search:  113  using fuzzy match
Fuzzy matching applied in  0.9886458953221638  minutes
number of terms missing after fuzzy match:  77
26.10169491525424  % NOT found
 Overall processing in  0.9886657079060872  minutes
 </pre>
 ## Tests
<pre> 
 python tests.py 
295  test results entries
77 With no results
26.1  % not found: 
On the %  73.9 found: 
Accuracy:  0.7981651376146789
Precision:  0.8046511627906977
Recall:  0.8009259259259259
F-measure 0.8027842227378191
 </pre>
