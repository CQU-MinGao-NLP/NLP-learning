import sys
sys.path.append("..") 

import torch
import torch.nn as nn
import torch.optim as optim
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

class NNLM_predict_N_word(Interface.Interface):
    def __init__(self, input_data = ["i like dog i like coffee", "i like coffee i like coffee", "i hate milk i like coffee", "i like coffee i like coffee", "i like chongqing i like coffee"],\
                 n_step = 5, n_hidden = 3, m = 3, lr = 0.001):
        #super(Interface, self).__init__()
        self.input_data = input_data
        self.m = m
        self.n_step = n_step
        self.n_hidden = n_hidden
        self.lr = lr

    # 控制流程
    def process(self):
        self.data_process()
        self.update_parameters()
        self.make_batch(self.input_data)
        self.model()
        self.optimization()
        self.train()
        self.predict()
        self.test()
    
    def update_parameters(self):
        self.parameters_name_list = ['n_class', 'm', 'n_step', 'n_hidden','lr']
        self.parameters_list = [self.n_class, self.m, self.n_step, self.n_hidden, self.lr]
        parameters_int_list = ['n_class', 'm', 'n_step', 'n_hidden'] # 输入为int
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
                

    # 将数据分为前N-1个词（input_batch）和所需预测的第N个词（target_batch）
    def make_batch(self, input_data):
        self.input_batch = []
        self.target_batch = []
        for sen in input_data:
            word = sen.split() # space tokenizer
            input = [self.word_dict[n] for n in word[:-1]] # create (1~n-1) as input
            target = self.word_dict[word[-1]] # create (n) as target, We usually call this 'casual language model'
            self.input_batch.append(input)
            self.target_batch.append(target)
        print("input_batch:")
        print(self.input_batch)
        print("target_batch:")
        print(self.target_batch)

    def data_process(self):
        # 得到词典，获取词典数目
        self.word_dict, self.number_dict, self.n_class = get_dictionary_and_num(self.input_data)


    def model(self):
        self.NNLM_model = NNLM(self.n_class, self.n_step, self.n_hidden, self.m)


    def optimization(self):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.NNLM_model.parameters(), lr=self.lr)

    def train(self):
        print('start train!')
        self.tensor_input_batch = torch.LongTensor(self.input_batch)
        self.tensor_target_batch = torch.LongTensor(self.target_batch)

        # Training
        for epoch in range(5000):
            self.optimizer.zero_grad()
            output = self.NNLM_model(self.tensor_input_batch)

            # output : [batch_size, n_class], target_batch : [batch_size]
            loss = self.criterion(output, self.tensor_target_batch)
            if (epoch + 1) % 1000 == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

            loss.backward()
            self.optimizer.step()
    def predict(self):
        # Predict
        print('start predict!')
        self.predict = self.NNLM_model(self.tensor_input_batch).data.max(1, keepdim=True)[1]
    
    def test(self):
        print([sen.split()[:self.n_step] for sen in self.input_data], '->', [self.number_dict[n.item()] for n in self.predict.squeeze()])