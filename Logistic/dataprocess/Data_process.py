import collections
import re
import sys
import nltk
import jieba
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

sys.path.append("..")


'''
Function: 
    [English] transfer input into word sequence
Input: 
    a list of sentence, and each element is a sentence
    (e.g. ["I love China!","We are family!"])
Output: 
    a list of single words, and each element is a sequence of words from a sentence
    (e.g. [['I', 'love', 'China', '!'], ['We', 'are', 'family', '!']])
'''
def en_tokenizer(input):
    text = []
    for line in input:
        temp = nltk.word_tokenize(line)
        text.append(temp)
    return text


'''
Function: 
    [Chinese] transfer input into word sequence
Input: 
    a list of sentence, and each element is a sentence
    (e.g. ["我爱中国","张帅好帅"])
Output: 
    a list of single words, and each element is a sequence of words from a sentence
    (e.g. [['我', '爱', '中国'], ['张帅', '好帅']])
'''
def cn_tokenizer(input):
    text = []
    for line in input:
        temp = list(jieba.cut(line, cut_all=False))
        text.append(temp)
    return text


'''
Function: 
    [English] delete stopwords from input
Input: 
    a list of sentence, and each element is a sentence
    (e.g. ["I love China!","We are family!"])
Output: 
    a list of single words, and each element is a sequence of words from a sentence except stopwords
    (e.g. [['I', 'love', 'China', '!'], ['We', 'family', '!']])
'''
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


'''
Function: 
    [Chinese] delete stopwords from input
Input: 
    1. a list of sentence, and each element is a sentence
    (e.g. ["我爱中国","张帅好帅"])
    2. a path of Chinese stopwords file
Output: 
    a list of single words, and each element is a sequence of words from a sentence except stopwords
    (e.g. [['爱', '中', '国'], ['张', '帅', '好', '帅']])
'''
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


'''
Function: 
    find the low-frequency words in the input and get a dictionary of low-frequency words
Input: 
    1. a list of single words, and each element is a sequence of words from a sentence
    (e.g. [['I', 'love', 'China', '!'], ['We', 'love', 'you', '!']])
    2. threshold of frequency n, and the default is 5
    (e.g. n = 2)
Output: 
    1. a dictionary of words and their frequencies
    (e.g. {'I': 1, 'China': 1, 'We': 1, 'you': 1})
    2. a list of words from dictionary
    (e.g. dict_keys(['I', 'China', 'We', 'you']))
'''
def lowfrequency(input_tokens, n=5):
    counter = collections.Counter([tk for st in input_tokens for tk in st])
    counter = dict(filter(lambda x: x[1] < n, counter.items()))
    return counter, counter.keys()


'''
Function: 
    find the high-frequency words in the input and get a dictionary of high-frequency words
Input: 
    1. a list of single words, and each element is a sequence of words from a sentence
    (e.g. [['I', 'love', 'China', '!'], ['We', 'love', 'you', '!']])
    2. threshold of frequency n, and the default is 5
    (e.g. n = 2)
Output: 
    1. a dictionary of words and their frequencies
    (e.g. {'love': 2, '!': 2})
    2. a list of words from dictionary
    (e.g. dict_keys(['love', '!']))
'''
def highfrequency(input_tokens, n=5):
    counter = collections.Counter([tk for st in input_tokens for tk in st])
    counter = dict(filter(lambda x: x[1] >= n, counter.items()))
    return counter, counter.keys()


'''
Function: 
    filter low-frequency words from input
Input: 
    1. a list of sentence, and each element is a sentence
    (e.g. ["I love China!","We love you!"])
    2. threshold of frequency n, and the default is 5
    (e.g. n = 2)
Output: 
    a list of single words, and each element is a sequence of words from a sentence except low-frequency words
    (e.g. [['love', '!'], ['love', '!']])
'''
def filter_lowfrequency(input, n=5):
    input_tokens = en_tokenizer(input)
    lowfre_list = lowfrequency(input_tokens,n)[1]
    for word in lowfre_list:
        for line in input_tokens:
            while word in line:
                line.remove(word)
    return input_tokens


'''
Function: 
    filter HTML tags from input
Input: 
    a list of sentence, and each element is a sentence
    (e.g. ["I love China!","<html> We love you! </html>"])
Output: 
    a list of sentence, and each element is sentence without HTML tags
    (e.g. ['I love China!', ' We love you! '])
'''
def filter_html(input):
    text = []
    for line in input:
        pattern = re.compile(r'<[^>]+>', re.S)
        temp = pattern.sub('', line)
        text.append(temp)
    return text



def lemmatization(self):
    pass


'''
Function: 
    restore the form of the words in the sentence
Input: 
    a list of sentence, and each element is a sentence
    (e.g. ["I like cats!","She liked dogs!"])
Output: 
    a list of single words, and each element is a sequence of words with normal form from a sentence 
    (e.g. [['i', 'like', 'cat', '!'], ['she', 'like', 'dog', '!']])
'''
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


'''
Function: 
    read input from a file
Input: 
    1. a path of file
    2. encoding
Output: 
    a list of sentence, and each element is a sentence
    (e.g. ["I love China!","We are family!"])
'''
def read_file(path, encoding="UTF-8"):
    with open(path, encoding=encoding) as f:
        text = f.readlines()
    return text


'''
Function: 
    save data into a file
Input: 
    1. a list of sentence, and each element is a sentence
    (e.g. ["I love China!","We are family!"])
    1. a path of file
    2. encoding
Output: 
    a file that store data
'''
def save_file(data, path, encoding="UTF-8"):
    with open(path, 'w', encoding=encoding,) as f:
        for line in data:
            for word in line:
                f.write(word)
                f.write(' ')
            f.write('\n')


'''
Function: 
    save dictionary into a file
Input: 
    1. a dictionary
    (e.g. {'0': 'pierre', '1': '<unk>', '2': 'N', '3': 'years'})
    2. a path of file you want to save
Output: 
    a file that store dictionary
'''
def dict_save(data, path):
    with open(path, 'w', encoding="UTF-8") as f:
        for key, value in data.items():
            f.write(str(key))
            f.write(" ")
            f.write(str(value))
            f.write('\n')


'''
Function: 
    read dictionary from a file
Input: 
    a path of file
Output: 
    a dictionary of words and their indexes
    (e.g. {'0': 'pierre', '1': '<unk>', '2': 'N', '3': 'years'})
'''
def dict_load(path):
    with open(path, encoding="UTF-8") as f:
        data = f.readlines()
    dict = {}
    for line in data:
        line = line.strip().split(" ")
        dict[line[0]] = line[1]

    return dict




