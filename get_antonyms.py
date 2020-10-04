#!/usr/bin/env python3

import inspect
import json
import pickle
import re
import time
from collections import defaultdict
from operator import itemgetter

import nltk
import requests
from bs4 import BeautifulSoup
from nltk.corpus import wordnet

'''
Antonym Dict = {
    WORD_1: {
        ORIGIN_1: [ANTONYM_1, ..., ANTONYM_n], 
        ...,
        ORIGIN_j: [ANTONYM_1, ..., ANTONYM_n]
    }, 
    ...,
     WORD_i: {
        ORIGIN_1: [ANTONYM_1, ..., ANTONYM_n], 
        ...,
        ORIGIN_j: [ANTONYM_1, ..., ANTONYM_n]
    },
}

ORIGINs:
    - WN (Antonyms found from WordNet)
    - TS (Antonyms found from Thesaurus)
    - PTS (Antonyms found from PowerThesaurus)
    - SYN (Synonyms found from WordNet, Thesaurus and PowerThesaurus)
'''

def word_net(word):
    '''
    Treat synsets of words as separate
    '''
    for syn in wordnet.synsets(word):
        synonyms, antonyms = [], []
        for l in syn.lemmas(): 
            synonyms.append(l.name()) 
            if l.antonyms(): 
                antonyms.append(l.antonyms()[0].name()) 
        # word_dict[syn]["synonyms"] = synonyms
        # word_dict[syn]["antonyms"] = antonyms
        # Choose which synsets pass through


def word_net_collapsed(word):
    '''
    Collapse synsets of the same word
    '''
    synonyms, antonyms = [], []
    for idx, syn in enumerate(wordnet.synsets(word)):
        if idx > 3:
            break
        for l in syn.lemmas():
            if l.name() not in synonyms: 
                synonyms.append(l.name()) 
            if l.antonyms():
                if len(l.antonyms()) > 1:
                    for ant in l.antonyms():
                        antonyms.append(ant.name())
                elif l.antonyms()[0].name() not in antonyms:
                    antonyms.append(l.antonyms()[0].name()) 
    return antonyms


def power_thesaurus(word, relation="antonyms", inspect=False):
    '''
    Scrape https://www.powerthesaurus.org for antonyms
    Relations: 
        synonyms :: Not-implemented 
        antonyms :: Works
        definitions :: Not-implemented 
        examples :: Not-implemented 
    '''
    url = "https://www.powerthesaurus.org/%s/%s" % (word, relation)
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}

    try:
        soup = BeautifulSoup(requests.get(url, headers=headers).text, 'html.parser')
    except Exception as e:
        print(f"The following error ({e}) occured for {url}, sleeping for 10 minutes ...")
        time.sleep(600)
        power_thesaurus(word)

    antonyms = []
    # class of antonym seems to change for PowerTheSaurus occasionally, therefore us regex to get the correct antonyms  
    regex = r"^/\w*/antonyms$" 
    for link in soup.find_all('a', href=True):
        if len(antonyms) > 9:
            break
        match = re.match(regex, link.attrs['href'])
        if match:
            ant = link.text if (link.text != word and link.text != "antonyms") else False
            if ant:
                antonyms.append(ant)

    if inspect:
        print(antonyms)
    
    return antonyms
    

def thesaurus(word, inspect=False):
    '''
    Scrape https://www.thesaurus.com/ for antonyms
    '''
    url = "https://www.thesaurus.com/browse/%s?s=t" % (word)
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}

    try:
        soup = BeautifulSoup(requests.get(url, headers=headers).text, 'html.parser')
    except Exception as e:
        print(f"The following error ({e}) occured for {url}, sleeping for 10 minutes ...")
        time.sleep(600)
        thesaurus(word)

    antonyms = []
    for idx, link in enumerate(soup.find_all('a', attrs={"css-lqr09m-ItemAnchor etbu2a31"}, href=True)):
        if len(antonyms) > 9:
            break
        antonyms.append(link.text)    

    if inspect:
        print(antonyms)    

    return antonyms


def inspect_antonym_dict(all_words=False):
    '''
    Inspect the antonym dict
        See if it is created
        See how many words it contains
        See the individual words and the amount of antonyms from each source
        See if any words lack antonyms from either thesauri
    '''
    try:
        with open("antonym_dict.pickle", "rb") as handle:
            antonym_dict = pickle.load(handle)
        print(f"Succesfully loaded antonym dict (contains {len(antonym_dict)} words!)")
    except FileNotFoundError as e:
        print("Could not find antonym dict, created empty dict!")
        antonym_dict = defaultdict(dd_module)

    if all_words:
        print("Word\tWordNet\t\tTheSaurus\tPowerTheSaurus")
        for word, antonym_sources in antonym_dict.items():
            l1 = []
            for source, antonyms in antonym_dict[word].items():
                l1.append((source, len(antonyms)))
            l1 = "\t".join(str(i) for i in l1)
            print(f"{word}\t{l1}")
    else:
        # Check if there are any words with 0 antonyms from TS or PTS
        ts_list, pts_list = [], [] # Save list of these words to retry antonym extraction process
        for word, antonym_sources in antonym_dict.items():
            for source, antonyms in antonym_dict[word].items():
                if (source == "TS" or source == "PTS") and len(antonyms) == 0:
                    print(f"{word}\t{source}\t{len(antonyms)}")
                    if source == "TS":
                        ts_list.append(word)
                    elif source == "PTS":
                        pts_list.append(word)
        return ts_list, pts_list


def dd_module():
    '''
    Module-level function (substitute to lambda function) for pickling
    '''
    return defaultdict(list)


def save_dict(antonym_dict):
    '''
    Save the antonym dict
    '''
    with open("antonym_dict.pickle", "wb") as handle:
        pickle.dump(antonym_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def calculate_antonym_score(antonyms, source, calculations, modifier=1):
    '''
    Calculate which antonym is the final best fitting antonym for a word
    '''
    if antonyms[source] != []:
        for idx, ant in enumerate(antonyms[source]):
            # WordNet antonyms are sparse, ensure they weigh similiar to the thesauri
            if source == "WN":
                modifier = (len(antonyms["TS"])+len(antonyms["PTS"]))//2
            score = (len(antonyms[source]) - idx)*modifier
            calculations[ant] += score
    return calculations


def get_antonym(word, inspect=False):
    '''
    Substitute a word with its antonym
    '''
    # Open or create antonym dict
    try:
        with open("antonym_dict.pickle", "rb") as handle:
            antonym_dict = pickle.load(handle)
    except FileNotFoundError as e:
        print("Could not find antonym dict, created empty dict!")
        antonym_dict = defaultdict(dd_module)

    # Check if antonyms already found for this word
    if antonym_dict[word] and antonym_dict[word]!={}:
        antonyms = antonym_dict[word]
        if antonym_dict[word]["WN"] == []:
            pass
        if  antonym_dict[word]["TS"] == []:
            ts = thesaurus(word)
            antonym_dict[word]["TS"] += ts
            antonyms = antonym_dict[word]
            save_dict(antonym_dict)
        if  antonym_dict[word]["PTS"] == []:
            pts = thesaurus(word)
            antonym_dict[word]["PTS"] += pts
            antonyms = antonym_dict[word]
            save_dict(antonym_dict)
    else:
        wn = word_net_collapsed(word)
        ts = thesaurus(word)
        pts = power_thesaurus(word)
        antonym_dict[word]["WN"] += wn
        antonym_dict[word]["TS"] += ts
        antonym_dict[word]["PTS"] += pts 

        # Save the new antonym dict
        save_dict(antonym_dict)
        # Retrieve the antonyms
        antonyms = antonym_dict[word]

    # Logic to define the final antonym    
    antonym =  ""
    calculations = defaultdict(lambda: 0) # set all possible antonyms to have a sore of zero
    sources = [("WN", 1), ("TS", 1), ("PTS", 1)] # list of sources and how much they weigh

    for src in sources:
        calculations = calculate_antonym_score(antonyms, src[0], calculations, modifier=src[1])

    top_antonyms = sorted(calculations.items(), key=itemgetter(1), reverse=True) # [(ant_1, freq_1), ..., (ant_i, freq_i)]

    if top_antonyms == []:
        antonym = f"not {word}"
        if inspect:
            print(f"Couldn't find any antonyms for '{word}', defaulted to '{antonym}'")
    else:
        antonym = top_antonyms[0][0]

    # print(f"Word = {word}\tAntonym = {antonym}")
    return antonym


def words_to_add():
    '''
    Pre-add some words to the antonym dict
    '''
    with open("progress_dict.pickle", "rb") as handle:
        d = pickle.load(handle)
    # Tags: {'NUM', 'PART', 'PRON', 'AUX', 'NOUN', 'CCONJ', 
    #        'ADJ', 'ADV', 'DET', 'INTJ', 'PROPN', 'ADP', 
    #        'VERB', 'X', 'SYM', 'PUNCT'}
    to_add = set()
    for k, v in d.items():
        words, pos_tags = k.split(), v.split()
        for word, tag in zip(words, pos_tags):
            if tag == "ADV":
                to_add.add(word)
    return to_add


def main():
    # word_net("warm")   
    # word_net_collapsed("warm")
    # power_thesaurus("warm", inspect=True)
    # thesaurus("warm", inspect=True)

    # inspect_antonym_dict(all_words=False)

    # words = words_to_add()
    # for w in words::
    #     get_antonym(w.lower())
    
    pass

if __name__ == '__main__':
    main()
