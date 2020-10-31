import sys
sys.path.append("..") 

import torch
import torch.nn as nn
import torch.optim as optim
import numpy
from Interface.NNLM_predict_next_word import NNLM_predict_N_word
from Interface.textCNN_classify import textCNN_classify
from Interface.transformer_translate import transformer_translate
from Interface.textRNN_classify import textRNN_classify
from Interface.word2vec import word2vec
#from Interface.FastText_classify_text import FastText_classify_text
#from Interface.Seq2seq_translate_text import Seq2seq_translate_text
from Interface.Bert_premodel_for_NLP import Bert_premodel_for_NLP

# 开始界面
def start():
    print('*'*80)
    print('Here is Help more people, natural language process (HMP-NLP)')
    print('This system has two ways to operate, we automatically use the system way, if you would like to use it by yourself, please contact us accordding to ReadMe')
    print('Members: Wang zongwei, Zhang Shuai, Wang Jia')
    print('Mentor: Gao Min')
    print('Organization: Chongqing University')

# 选择所需系统功能
def choose_system():
    print('*'*80)
    print('Our system are seperated into three parts:')
    print(' 1. Data Process (this part)')
    print(' 2. Use Trained Model to Predict (Simple way) (this part)')
    print(' 3. Train Model by Yourself (Professional way) (this part)')
    print('Which part do you want to enter?')
    try:
        input_system = int(input())
    except KeyError:
        print("Error input")
        exit(-1)
    return input_system

# 选择所需的接口
def choose():
    print('*'*80)
    print('There are some basic models to choose, please choose one')
    print('(example: if you input 1 with <enter>, you will run NNLM)')
    print('1. NNLM_predict_next_word(sample)    2. textCNN_classify           3. transformer_translate')
    print('4. textRNN_classify                  5. word2vec                   6. ELMo')
    print('7. FastText_classify_text            8. Seq2seq_translate_text     9. Bert_premodel_for_NLP')
    print('*'*80)
    try:
        number = int(input())
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
            print("Enter Data Process")

        elif system_number == 2:
            print("Enter P2")
        elif system_number == 3:
            number = choose()
            if number == 1:
                test = NNLM_predict_N_word()
            elif number == 2:
                test = textCNN_classify()
            elif number == 3:
                test = transformer_translate()
            elif number == 4:
                test = textRNN_classify()
            elif number == 5:
                test = word2vec()
            elif number == 7:
                test = FastText_classify_text()
            elif number == 8:
                test = Seq2seq_translate_text()
            elif number == 9:
                test = Bert_premodel_for_NLP()
            else:
                pass
            test.process()
        else:
            print("wrong input")
            print('*'*80)
        print("the process of this opration is over, do you want exit our system? (yes/no)")
        try:
            input_end_looper = str(input())
        except KeyError:
            print("Error input!")
            exit(-1)
        if input_end_looper == 'yes':
            break
        elif input_end_looper == 'no':
            pass
        else:
            print("Error input")
            exit(-1)
    end()
