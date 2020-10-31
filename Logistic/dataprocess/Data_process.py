import collections
import re

import nltk
import jieba
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

def en_tokenizer(input):
    text = []
    for line in input:
        temp = nltk.word_tokenize(line)
        text.append(temp)
    return text

def cn_tokenizer(input):
    text = []
    for line in input:
        temp = jieba.cut(line, cut_all=False)
        text.append(temp)
    return text

def en_stopwords(input):
    text = []
    temp = []
    for line in input:
        for word in line:
            if word in stopwords:
                continue
            else:
                temp.append(word)
        text.append(temp)
    return text

def cn_stopwords(input, path=""):
    with open(path, encoding='GB18030') as f:
        stopwords_f = f.readlines()

    stopwords = []
    for line in stopwords_f:
        stopwords.append(line.strip())

    text = []
    temp = []
    for line in input:
        for word in line:
            if word in stopwords:
                continue
            else:
                temp.append(word)
        text.append(temp)
    return text


def lowfrequency(input_tokens, n):
    counter = collections.Counter([tk for st in input_tokens for tk in st])
    counter = dict(filter(lambda x: x[1] < n, counter.items()))
    return counter.keys()

def filter_lowfrequency(input_tokens, lowfre_list):
    for word in lowfre_list:
        for line in input_tokens:
            while word in line:
                line.remove(word)
    return input_tokens


def html(input):
    text = []
    for line in input:
        pattern = re.compile(r'<[^>]+>', re.S)
        temp = pattern.sub('', line)
        text.append(temp)
    return text

def lemmatization(self):
    pass

def stemming(input):
    snowball_stemmer = SnowballStemmer("english")
    text = []
    temp = []
    for line in input:
        for word in line:
            snowball_stemmer.stem(word)
            temp.append(snowball_stemmer)
        text.append(temp)
    return text


print(en_tokenizer("I love china!"))


