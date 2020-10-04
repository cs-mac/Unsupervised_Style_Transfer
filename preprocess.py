#!/usr/bin/env python3

import glob
import pickle
import random
import string
import re

import pandas as pd
from sklearn.utils import shuffle
from tqdm import tqdm

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.spellcorrect import SpellCorrector
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

tqdm.pandas()

sp = SpellCorrector(corpus="twitter")
text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=[],
    fix_html=True,  # fix HTML tokens
    
    # corpus from which the word statistics are going to be used 
    # for word segmentation 
    segmenter="english", 
    
    unpack_hashtags=False,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=True,  # spell correction for elongated words
    
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=False).tokenize,
    
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[]
)


def datareader(dir, filetype="txt", engine="c"):
    '''
    Read in data from file into a pandas DataFrame
    '''
    frames = []
    
    if not dir.endswith("/"):
        dir = dir+"/"

    for file in glob.glob(dir+"*."+filetype):
        filename = file.split(dir)[1] 
        data = []
        sep_f = pd.DataFrame()
        with open(dir+filename, "r") as f:
            for line in f:
                line = line.strip()
                data.append(line)        
        sep_f["words"] = data
        sep_f["categories"] = 0 if "neg" in filename else 1
        frames.append(sep_f)
    return pd.concat(frames)


def lower(sentence):
    '''
    Change to normlisation func
    '''
    return sentence.lower()


def remove_dots(sentence):
    '''
    Ensure there are no '.' (dots) repeating, since these are 
    considered separate lines according to our HAN model,
    '''
    return sentence.translate(str.maketrans('', '', string.punctuation))


def spell_correct(sentence):
    '''
    Correct spelling mistakes
    '''
    return " ".join(sp.correct(word) for word in sentence.split())


def decontract(sentence):
    '''
    Unpack contractions (e.g. can't -> can not)
    '''
    # specific
    sentence = re.sub(r"won\'t", "will not", sentence)
    sentence = re.sub(r"can\'t", "can not", sentence) #instead of cannot
    sentence = re.sub(r"shan\'t", "shall not", sentence)
    sentence = re.sub(r"(i|I) ain\'t", "i am not", sentence)
    sentence = re.sub(r"(h|H)e ain\'t", "he is not", sentence)
    sentence = re.sub(r"(s|S)e ain\'t", "she is not", sentence)
    sentence = re.sub(r"(w|W)e ain\'t", "we are not", sentence)

    # general
    sentence = re.sub(r"n\'t", " not", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"\'s", " is", sentence)
    sentence = re.sub(r"\'d", " would", sentence)
    sentence = re.sub(r"\'ll", " will", sentence)
    sentence = re.sub(r"\'t", " not", sentence)
    sentence = re.sub(r"\'ve", " have", sentence)
    sentence = re.sub(r"\'m", " am", sentence)
    return sentence    


def normalise(sentence):
    '''
    ::  Normalize webaddresses, e-mail adresses, percentages, valuta, phone-numbers, time data, dates, and numbers to 
        ['url', 'email', 'percent', 'money', 'phone', 'time', 'date', 'number']
    ::  Fix any HTML tags left in data
    ::  Segment words (e.g. whatisthis -> what is this)
    ::  Spelling correction (e.g. fauld -> fault, looooool -> lol)
    ::  Basic tokenization
    ::  Emoticon substitution
    '''
    return " ".join(text_processor.pre_process_doc(sentence))


def main():
    data = datareader("data/sentiment_reviews/")
    data.reset_index(inplace=True, drop=True)

    # Some pre-processing steps
    data["words"] = data["words"].apply(decontract)
    data["words"] = data["words"].progress_apply(normalise)
    data["words"] = data["words"].apply(lower)
    data["words"] = data["words"].apply(remove_dots)
    data["words"] = data["words"].progress_apply(spell_correct)

    for i in range(3):
        data = shuffle(data)
        data.reset_index(inplace=True, drop=True)

    # Create balanced: train (600k), validation (100k) and test sets (100k)
    data["is_which_set"] = "train"
    data.is_which_set.iloc[data[data["categories"]==1].is_which_set.index[300000:350000]] = "val"
    data.is_which_set.iloc[data[data["categories"]==1].is_which_set.index[350000:]] = "test"
    data.is_which_set.iloc[data[data["categories"]==0].is_which_set.index[300000:350000]] = "val"
    data.is_which_set.iloc[data[data["categories"]==0].is_which_set.index[350000:]] = "test"

    data.to_pickle('df_all.pkl')

    # data = pd.read_pickle('df_all.pkl')
    # pd.set_option('display.max_colwidth', -1) # show more of pandas dataframe
    # print(data.head())

if __name__ == '__main__':
    main()
