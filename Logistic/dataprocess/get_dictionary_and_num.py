'''
得到词典，获取词典数目
输入格式为 list
["i like dog", "i love coffee", "i hate milk", "i like coffee", "i love chongqing"]
输出为： 
dict(): key是word， value是index
dict(): key是index， value是word
int(): 词典的word数
'''

def get_dictionary_and_num(data):
    word_list = list(set(tk for st in data for tk in st))
    word_dict = {w: i for i, w in enumerate(word_list)}
    number_dict = {i: w for i, w in enumerate(word_list)}
    n_class = len(word_dict)  # number of Vocabulary
    return word_dict, number_dict, n_class

