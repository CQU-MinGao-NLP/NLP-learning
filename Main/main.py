import os
import re
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
from Interface.model_test import Model_test
#from Interface.FastText_classify_text import FastText_classify_text
#from Interface.Seq2seq_translate_text import Seq2seq_translate_text
from Interface.Bert_premodel_for_NLP import Bert_premodel_for_NLP
from Interface.Data_process_controler import Data_process_controler

# 判断循环(包括合法性、选项存在与否、文件存在与否）
def loop_legal(input_ori, type, max = 9, path = ""):
    # 判断输入格式是否有误
    while (input_legal(input_ori, type) == False):
        if type == 1:
            print("[ERROR] Your input format is incorrect, please re-enter (example : 1) :")
        elif type == 2:
            print("[ERROR] Your input format is incorrect, please re-enter (example : sample.txt) :")
        input_now = input()
        if input_legal(input_now, type) == True:
            input_ori = input_now
            break
        else:
            continue
    # 判断是否有该选项
    if type == 1:
        while (int(input_ori) > max or int(input_ori) <=int(0)):
            print("[ERROR] this option does not exist!, please re-enter  (example : 1) :")
            input_now = input()
            if input_legal(input_now, type) == True:
                if (int(input_now) > max or int(input_now) <=int(0)):
                    continue
                else:
                    input_ori = input_now
                    break
        return int(input_ori)
    # 判断是否存在该文件
    elif type == 2:
        while os.path.exists(path + input_ori) == False:
            print("[ERROR] this file does not exist!, please re-enter  (example : sample.txt) :")
            input_now = input()
            if input_legal(input_now, type) == True:
                if os.path.exists(path + input_now) == False:
                    continue
                else:
                    input_ori = input_now
                    break
        return input_ori

# 判断输入合法性
def input_legal(input_ori, type):
    legal = False
    # 当需要输入int类型时
    if type == 1:
        matchObj = re.match(r"[0-9]", input_ori)
        if matchObj != None:
            legal = True
            return legal
        else:
            return legal

    # 当需要输入文件名
    elif type == 2:
        matchObj = re.match(r'\w+.txt', input_ori)
        if matchObj != None:
            legal = True
            return legal
        else:
            return legal

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
        input_system = loop_legal(input(), 1, max=3)
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
    print("4. cn_stopwords  5. lowfrequency     6. highfrequency")
    print("7. filter_lowfrequency  8. filter_html     9. stemming")
    enter_num = loop_legal(input(), 1, max=9)
    print("please print the filename you want to process: (example: sample.txt)")
    filename = loop_legal(input(), 2, path=rawdata_root)
    controler = Data_process_controler(filename, enter_num)
    controler.process()
    print('*'*80)

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
        number = loop_legal(input(), 1, max=9)
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
            testdata_root = "../Data/test_data/"
            print('*' * 80)
            print("Use Trained Model to Predict!")
            print("please input the type of task:")
            print("1: embedding    2: classify")
            number = int(loop_legal(input(), 1, max=2))
            if number == 1:
                print("please input model id:")
                print("1: word2vec")
                number = int(loop_legal(input(), 1, max=1))
                if number == 1:
                    print('please input the filename:')
                    file = loop_legal(input(), 2, path=testdata_root)
                    model = Model_test(1, 1, filename=file)
                    model.process()


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
                file = input('please input the filename\n')
                test = word2vec(filename=file)
            elif number == 7:
                test = FastText_classify_text()
            elif number == 8:
                test = Seq2seq_translate_text()
            elif number == 9:
                test = Bert_premodel_for_NLP()
            else:
                pass
            test.process()
        #else:
        #    print("wrong input")
        #    print('*'*80)
        print("the process of this operation is over, do you want exit our system? (yes/no)")
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
