import sys
import os
import fasttext

sys.path.append("..")

import torch
import torch.nn as nn
import torch.optim as optim
import numpy
from Interface import Interface
import warnings
from Logistic.dataprocess.Data_process import *
warnings.filterwarnings("ignore")

'''
功能：
    使用FastText模型，将一段文本进行快速分类
输入（可调整的参数）：
    [1]intput: 数据文件，其中每一行为  
           __label__Shares , 中铁 物资 最快 今年底 A H股 上市 募资 120 亿 陈 姗姗 募集 资金 100 亿到 120 亿元 央企 中国...... 
    [2]lr: 学习率       
    [3]dim： 词向量维度          
    [4]epoch： 训练次数
    [5]word_ngram:  做ngram的窗口大小
    [6]loss:  损失函数类型
输出：
    给测试集的每段文本进行分类
'''

MODEL_ROOT = "../Data/model/fasttext_classify/"
DATA_ROOT = "../Data/train_data/"

class FastText_classify_text(Interface.Interface):
    def __init__(self, filename, lr=0.2, dim=100, epoch=500, word_ngrams=4, loss='softmax'):
        self.filename = filename
        self.lr = lr
        self.dim = dim
        self.epoch = epoch
        self.word_ngrams = word_ngrams
        self.loss = loss

    # 控制流程
    def process(self):
        print("loading data...")
        self.data_process()
        print("loading data succeed!")
        self.update_parameters()
        self.model()
        # self.predict()

    def data_process(self):
        assert self.filename in os.listdir(DATA_ROOT)

        
        data = read_file(DATA_ROOT+self.filename)
        
        print(DATA_ROOT + self.filename[:-4] + '_Fasttext.txt')
        with open(DATA_ROOT + self.filename[:-4] + '_Fasttext.txt', 'w', encoding="UTF-8") as f:
            for line in data:
                line = line.strip().split(' ')
                f.write('__label__')
                f.write(line[0])
                f.write(' , ')
                for i in line[1:]:
                    f.write(i)
                    f.write(' ')
                f.write('\n')



    def update_parameters(self):
        self.parameters_name_list = ['lr', 'dim', 'epoch', 'word_ngrams']
        self.parameters_list = [self.lr, self.dim, self.epoch, self.word_ngrams]
        parameters_int_list = ['dim', 'epoch', 'word_ngrams'] # 输入为int
        parameters_float_list = ['lr'] # 输入为float

        while 1:
            print("Model parameters are:")
            for i in range(len(self.parameters_list)):
                print("[{}] {}={}".format(i, str(self.parameters_name_list[i]), self.parameters_list[i]))
            print("Do you want change model parameters?(yes/no)")
            try:
                input_choose = str(input())
            except KeyError:
                    print("Error num!")
                    exit(-1)
            if input_choose == 'yes':
                print('choose parameter you want, give the number of parameter')
                try:
                    input_number = int(input())
                except KeyError:
                    print("Error num!")
                    exit(-1)
                print("your choose {}, print the number you want change".format(self.parameters_name_list[input_number]))
                try:
                    while True:
                        parameter = input()
                        if self.parameters_name_list[input_number] in parameters_int_list:
                            parameter = int(parameter)
                            break
                        elif self.parameters_name_list[input_number] in parameters_float_list:
                            parameter = float(parameter)
                            break
                        else:
                            print("Error input, your input format is wrong!")   
                except KeyError:
                    print("Error num!")
                    exit(-1)
                self.parameters_list[input_number] = parameter
                print("update success!")
            elif input_choose == 'no':
                break
            else:
                print("wrong input, please input again！")
                pass


    def model(self):
        self.FastText_model = fasttext.train_supervised(DATA_ROOT + self.filename[:-4] + '_Fasttext.txt', lr=self.lr, dim=self.dim, epoch=self.epoch, word_ngrams=self.word_ngrams, loss=self.loss)
        if os.path.exists(MODEL_ROOT) == False:
            os.mkdir(MODEL_ROOT)
        self.FastText_model.save_model(MODEL_ROOT + 'fasttext_model_file.bin')


    def optimization(self):
        pass

    def train(self):
        pass

    def predict(self):
        classifier = fasttext.load_model('../model/fasttext_model_file.bin')
        print(classifier.predict("洪世贤 真 是 渣 得 明明白白, 刘亦菲 好 漂亮！"))
        f1 = open('../prediction/prediction.txt', 'w', encoding='utf-8')
        with open('../datasets/text-classify/text_classify_test.txt', encoding='utf-8') as fp:
            for line in fp.readlines():
                line = line.strip()
                if line == '':
                    continue
                f1.write(line + '\t#####\t' + classifier.predict([line])[0][0][0] + '\n')
        f1.close()
    
    def test(self):
        pass