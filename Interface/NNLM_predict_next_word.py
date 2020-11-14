import sys
import os

from Interface.loop_legal import loop_legal
from Logistic.dataprocess.Data_process import dict_save

sys.path.append("..")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import numpy
from Interface import Interface
from Logistic.dataprocess.get_dictionary_and_num import *
from Logistic.model.NNLM import NNLM
'''
功能：
    使用NNLM模型，利用前N-1个word预测第N个word
输入（可调整的参数）：
    [1]词库大小(n_class)     
    [2]转化的词向量大小(m)        
    [3]输入层神经元数(即词的滑动窗口容量, n_step)            
    [4]隐层神经元数量(n_hidden)
    [5]学习率(lr)
输出：
    针对给出的前N-1个word预测出的第N个word的list序列
'''

MODEL_ROOT = "../Data/model/NNLM_predict_next_word/"
DATA_ROOT = "../Data/train_data/"

class NNLM_predict_next_word(Interface.Interface):
    def __init__(self, filename, n_step=5, n_hidden=3, m=3, lr=0.001, batch_size=32, num_epochs=500):
        #super(Interface, self).__init__()
        self.input_data = []
        self.input = []
        self.target = []
        self.filename = filename
        self.m = m
        self.n_step = n_step
        self.n_hidden = n_hidden
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    # 控制流程
    def process(self):
        print("loading data...")
        self.data_process()
        print("loading data succeed!")
        self.update_parameters()
        self.make_batch()
        self.model()
        self.optimization()
        self.train()
        #self.predict()
        #self.test()

    def data_process(self):
        with open(DATA_ROOT + self.filename, 'r') as f:
            lines = f.readlines()
            raw_dataset = [st.split() for st in lines]

        for line in raw_dataset:
            length = len(line)
            if length < (self.n_step+1):
                continue
            for i in range(len(line)-self.n_step-1):
                self.input_data.append(line[i:(i+self.n_step+1)])
        # 得到词典，获取词典数目
        self.word_dict, self.number_dict, self.n_class = get_dictionary_and_num(self.input_data)
        if os.path.exists(MODEL_ROOT) == False:
            os.mkdir(MODEL_ROOT)
        dict_save(self.number_dict, MODEL_ROOT + "NNLM_predict_next_word_idx2token.txt")
        dict_save(self.word_dict, MODEL_ROOT + "NNLM_predict_next_word_token2idx.txt")
        self.deal_input_data()

    # 将数据分为前N-1个词（input）和所需预测的第N个词（target）
    def deal_input_data(self):
        for word in self.input_data:
            input = [self.word_dict[n] for n in word[:-1]] # create (1~n-1) as input
            target = self.word_dict[word[-1]] # create (n) as target, We usually call this 'casual language model'
            self.input.append(input)
            self.target.append(target)

    def update_parameters(self):
        self.parameters_name_list = ['m', 'n_step', 'n_hidden','lr','batch_size', 'num_epochs']
        self.parameters_list = [self.m, self.n_step, self.n_hidden, self.lr, self.batch_size, self.num_epochs]
        parameters_int_list = ['m', 'n_step', 'n_hidden', 'batch_size', 'num_epochs'] # 输入为int
        parameters_float_list = ['lr'] # 输入为float

        while 1:
            print("Model parameters are:")
            for i in range(len(self.parameters_list)):
                print("[{}] {}={}".format(i, str(self.parameters_name_list[i]), self.parameters_list[i]))
            print("Do you want change model parameters?(y/n)")
            try:
                input_choose = loop_legal(str.lower(input()), 3)
            except KeyError:
                    print("Error num!")
                    exit(-1)
            if input_choose == 'y':
                print('choose parameter you want, give the number of parameter')
                try:
                    input_number = loop_legal(input(), 1, max_value=len(self.parameters_list))
                except KeyError:
                    print("Error num!")
                    exit(-1)
                print("your choose {}, print the number you want change".format(self.parameters_name_list[input_number]))
                try:
                    while True:
                        if self.parameters_name_list[input_number] in parameters_int_list:
                            parameter = loop_legal(input(), 1, max_value=sys.maxsize)
                            parameter = int(parameter)
                            break
                        elif self.parameters_name_list[input_number] in parameters_float_list:
                            parameter = loop_legal(input(), 4, max_value=float(1.0))
                            parameter = float(parameter)
                            break
                        else:
                            print("Error input, your input format is wrong!")   
                except KeyError:
                    print("Error num!")
                    exit(-1)
                self.parameters_list[input_number] = parameter
                print("update success!")
            elif input_choose == 'n':
                break
            else:
                print("wrong input, please input again！")
                pass
                



    def make_batch(self):
        self.data_iter = Data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(self.input), torch.tensor(self.target)), self.batch_size, shuffle=True)

    def model(self):
        self.m, self.n_step, self.n_hidden, self.lr, self.batch_size, self.num_epochs = \
            self.parameters_list[0], self.parameters_list[1], self.parameters_list[2], self.parameters_list[3], \
            self.parameters_list[4], self.parameters_list[5]
        self.NNLM_model = NNLM(self.n_class, self.n_step, self.n_hidden, self.m)


    def optimization(self):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.NNLM_model.parameters(), lr=self.lr)

    def train(self):
        print('start train!')
        # Training
        print(self.parameters_list)
        print(self.num_epochs)
        for epoch in range(self.num_epochs):
            self.optimizer.zero_grad()
            for X, y in self.data_iter:
                output = self.NNLM_model(X)

                # output : [batch_size, n_class], target_batch : [batch_size]
                loss = self.criterion(output, y)
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

            loss.backward()
            self.optimizer.step()

        if os.path.exists(MODEL_ROOT) == False:
            os.mkdir(MODEL_ROOT)
        torch.save(self.NNLM_model, MODEL_ROOT + 'NNLM_predict_next_word.pkl')
        print("The model has been saved in " + MODEL_ROOT[3:] + 'NNLM_predict_next_word.pkl')

    def predict(self):
        # Predict
        print('start predict!')
        self.predict = self.NNLM_model(self.tensor_input_batch).data.max(1, keepdim=True)[1]
    
    def test(self):
        print([sen.split()[:self.n_step] for sen in self.input_data], '->', [self.number_dict[n.item()] for n in self.predict.squeeze()])