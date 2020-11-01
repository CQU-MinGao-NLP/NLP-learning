import collections
import re
import sys
import nltk
import jieba
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

sys.path.append("..")

def en_tokenizer(input):
    text = []
    for line in input:
        temp = nltk.word_tokenize(line)
        text.append(temp)
    return text

def cn_tokenizer(input):
    text = []
    for line in input:
        temp = list(jieba.cut(line, cut_all=False))
        text.append(temp)
    return text

def en_stopwords(input):
    text = []
    input = en_tokenizer(input)
    stwords = stopwords.words('english')
    for line in input:
        temp = []
        for word in line:
            if word in stwords:
                continue
            else:
                temp.append(word)
        text.append(temp)
    return text

def cn_stopwords(input, path="../Data/stopwords.txt"):
    with open(path, encoding='GB18030') as f:
        stopwords_f = f.readlines()

    stopwords = []
    for line in stopwords_f:
        stopwords.append(line.strip())

    text = []
    for line in input:
        temp = []
        for word in line:
            if word in stopwords:
                continue
            else:
                temp.append(word)
        text.append(temp)
    return text


def lowfrequency(input_tokens, n=5):
    counter = collections.Counter([tk for st in input_tokens for tk in st])
    counter = dict(filter(lambda x: x[1] < n, counter.items()))
    return counter, counter.keys()

def highfrequency(input_tokens, n=5):
    counter = collections.Counter([tk for st in input_tokens for tk in st])
    counter = dict(filter(lambda x: x[1] >= n, counter.items()))
    return counter, counter.keys()


def filter_lowfrequency(input, n):
    input_tokens = en_tokenizer(input)
    lowfre_list = lowfrequency(input_tokens,n)[1]
    for word in lowfre_list:
        for line in input_tokens:
            while word in line:
                line.remove(word)
    return input_tokens


def filter_html(input):
    text = []
    for line in input:
        pattern = re.compile(r'<[^>]+>', re.S)
        temp = pattern.sub('', line)
        text.append(temp)
    return text

def lemmatization(self):
    pass

def stemming(input):
    input = en_tokenizer(input)
    snowball_stemmer = SnowballStemmer("english")
    text = []
    for line in input:
        temp = []
        for word in line:
            temp.append(snowball_stemmer.stem(word))
        text.append(temp)
    return text


def read_file(path, encoding="UTF-8"):
    with open(path, encoding=encoding) as f:
        text = f.readlines()
    return text


def save_file(data, path, encoding="UTF-8"):
    with open(path, 'w', encoding=encoding,) as f:
        for line in data:
            for word in line:
                f.write(word)
                f.write(' ')
            f.write('\n')


def dict_save(data, path):
    with open(path, 'w', encoding="UTF-8") as f:
        for key, value in data.items():
            f.write(str(key))
            f.write(" ")
            f.write(str(value))
            f.write('\n')

def dict_load(path):
    with open(path, encoding="UTF-8") as f:
        data = f.readlines()
    dict = {}
    for line in data:
        line = line.strip().split(" ")
        dict[line[0]] = line[1]

    return dict


# print(en_tokenizer(["I love china!","Landw dnwqinv uwf"]))
# print(cn_tokenizer(["我爱中国","张帅好帅"]))
# print(stemming(["I love china cats!","ndw dnwi fnwi"]))
# text = read_file("../../Data/sample.txt")
# print(en_tokenizer(text))
# save_file(en_tokenizer(text))


