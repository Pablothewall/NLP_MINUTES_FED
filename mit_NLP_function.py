# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 12:16:59 2023

@author: l11420
"""
# pip install -U spacy
# python -m spacy download en_core_web_sm

import spacy

def spacy_npl(txt):

    #import en_core_web_sm
    #nlp = en_core_web_sm.load()
    # Load English tokenizer, tagger, parser and NER
    nlp = spacy.load("en_core_web_sm")
    
    
    
    # Process whole documents
    text = (txt)
    doc = nlp(text)
    
    # Analyze syntax
    #print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
    return [token.lemma_ for token in doc if token.pos_ == "VERB"]



    