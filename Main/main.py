import sys

sys.path.append("..")

import torch
import torch.nn as nn
import torch.optim as optim
import numpy
'''
from Interface.NNLM_predict_next_word import NNLM_predict_next_word
from Interface.textCNN_classify import textCNN_classify
from Interface.transformer_translate import transformer_translate
from Interface.textRNN_classify import textRNN_classify
from Interface.word2vec import word2vec
#from Interface.FastText_classify_text import FastText_classify_text
#from Interface.Seq2seq_translate_text import Seq2seq_translate_text
from Interface.Bert_premodel_for_NLP import Bert_premodel_for_NLP
'''

from Interface.model_test import Model_test
from Interface.Data_process_controler import Data_process_controler
from Interface.loop_legal import loop_legal


data_process_dict = {"1": "en_tokenizer",  "2": "cn_tokenizer", "3": "en_stopwords",
                     "4": "cn_stopwords",  "5": "filter_lowfrequency", "6": "filter_html",
                     "7": "stemming"}

train_model_dict = {"1": 'NNLM_predict_next_word', "2": 'textCNN_classify', "3": 'transformer_translate',
                    "4": 'textRNN_classify', "5": 'word2vec', "6": 'Seq2seq_translate_text',
                    "7": 'Bert_premodel_for_NLP', "8": 'FastText_classify_text'}

test_model_dict = {'1': ['embedding', {'1': 'word2vec', '2': 'elmo', "3": 'bert'}], '2':['classify', {"1": 'textRNN',"2": 'textCNN', "3": 'fasttext'}], '3':['translation', {"1": 'seq2seq', "2": 'transformer'}], '4': ['generation', {"1": 'NNLM'}]}

# 开始界面
def start():
    print('*'*80)
    print('Here is Help more people, natural language process (HMP-NLP)')
    print('This system has two ways to operate, we automatically use the system way, if you would like to use it by yourself, please contact us accordding to ReadMe')
    print('Members: Yinqiu Huang, Wang zongwei, Zhang Shuai, Wang Jia')
    print('Mentor: Gao Min')
    print('Organization: Chongqing University')

# 选择所需系统功能
def choose_system():
    print('*'*80)
    print('Our system are seperated into three parts:')
    print(' 1. Data Process ')
    print(' 2. Use Trained Model to Predict (Simple way) ')
    print(' 3. Train Model by Yourself (Professional way) ')
    print('Which part do you want to enter?')
    try:
        input_system = loop_legal(input(), 1, max_value=3)
    except KeyError:
        print("Error input")
        exit(-1)
    return input_system

# 选择数据处理功能
def data_process(rawdata_root):
    print('*'*80)
    print("Enter Data Process System!")
    print("please choose your operation: (the number of operation)")
    print("1. en_tokenizer  2. cn_tokenizer     3. en_stopwords")
    print("4. cn_stopwords  5. filter_lowfrequency     6. filter_html")
    print("7. stemming")
    enter_num = loop_legal(input(), 1, max_value=len(data_process_dict))
    print("please input the filename (example: sample.txt) you want to process, assert that your file is in \"Data/raw_data\": ")
    #filename = loop_legal(input(), 2, path=rawdata_root)
    filename = input()
    controler = Data_process_controler(filename, data_process_dict[str(enter_num)])
    controler.process()
    print('*'*80)

# 选择所需的接口
def train_choose():
    print('*'*80)
    print('There are some basic models to choose, please choose one')
    print('(example: if you input 1 with <enter>, you will run NNLM)')
    print('1. NNLM_predict_next_word(sample)    2. textCNN_classify           3. transformer_translate')
    print('4. textRNN_classify                  5. word2vec                   6. Seq2seq_translate_text')
    print('7. Bert_premodel_for_NLP             8. FastText_classify_text')
    print('*'*80)
    try:
        number = loop_legal(input(), 1, max_value=len(train_model_dict))
    except KeyError:
        print("Error num!")
        exit(-1)
    return number

# 结束界面
def end():
    print('*'*80)
    print('It is the ending! If you would like to use it again, please run again~')
    print('*'*80)

if __name__ == '__main__':
    start()
    while True:
        system_number = choose_system()
        if system_number == 1:
            rawdata_root = "../Data/raw_data/"
            data_process(rawdata_root)

        elif system_number == 2:
            test_data_root = "../Data/test_data/"
            print('*' * 80)
            print("Use Trained Model to Predict!")
            print("please input the type of task:")
            print(test_model_dict)
            # for key in test_model_dict:
            for key, value in test_model_dict.items():
                print(key + ': ' + value[0])
            task_id = int(loop_legal(input(), 1, max_value=len(test_model_dict.keys())))

            print("you choose " + test_model_dict[str(task_id)][0] + ", please input model id:")
            
            for key, value in test_model_dict[str(task_id)][1].items():
                print(key + ': ' + value)

            model_id = int(loop_legal(input(), 1, max_value=len(test_model_dict[str(task_id)][1])))
            
            file = loop_legal(input('please input the filename, assert that your file is in \"Data/test_data\" \n'), 2, path=test_data_root)
            model = Model_test(task_id, model_id, test_model_dict, filename=file)
            model.process()

        elif system_number == 3:
            train_data_root = "../Data/train_data/"
            number = train_choose()
            file = loop_legal(input('please input the filename, assert that your file is in \"Data/train_data\" \n'),
                              2, path=train_data_root)
            exec('from Interface.' + train_model_dict[str(number)] + ' import ' + train_model_dict[str(number)])
            if number == 1:
                test = NNLM_predict_next_word(filename=file)
            elif number == 2:
                test = textCNN_classify(filename=file)
            elif number == 3:
                test = transformer_translate(filename=file)
            elif number == 4:
                test = textRNN_classify(filename=file)
            elif number == 5:
                test = word2vec(filename=file)
            elif number == 6:
                test = Seq2seq_translate_text(filename=file)
            elif number == 7:
                test = Bert_premodel_for_NLP(filename=file)
            elif number == 8:
                test = FastText_classify_text(filename=file)
            else:
                pass
            test.process()
        #else:
        #    print("wrong input")
        #    print('*'*80)
        print("the process of this operation is over, do you want exit our system? (y/n)")
        try:
            input_end_looper = loop_legal(str.lower(input()), 3)
        except KeyError:
            print("Error input!")
            exit(-1)
        if input_end_looper == 'y':
            break
        elif input_end_looper == 'n':
            pass
        else:
            print("Error input")
            exit(-1)
    end()
