import sys

import fasttext

sys.path.append("..")

import torch
import torch.nn as nn
import torch.optim as optim
import numpy
from Interface import Interface
import warnings
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

class FastText_classify_text(Interface.Interface):
    def __init__(self, input = '../datasets/text-classify/text_classify_train.txt', lr=0.2, dim=100, epoch=500, word_ngrams=4, loss='softmax'):
        self.input = input
        self.lr = lr
        self.dim = dim
        self.epoch = epoch
        self.word_ngrams = word_ngrams
        self.loss = loss

    # 控制流程
    def process(self):
        self.model()
        self.predict()

    def data_process(self):
        pass


    def model(self):
        self.FastText_model = fasttext.train_supervised(self.input, lr=self.lr, dim=self.dim, epoch=self.epoch, word_ngrams=self.word_ngrams, loss=self.loss)
        self.FastText_model.save_model('../model/fasttext_model_file.bin')


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