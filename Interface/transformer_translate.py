import sys
sys.path.append("..") 

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Interface import Interface
from Logistic.dataprocess.get_dictionary_and_num import *
from Logistic.model.transformer import Transformer
'''
功能：
    使用transformer进行机器翻译
输入:
    [1]src_len = 5 # length of source 原文本的长度
    [2]tgt_len = 5 # length of target   目标文本的长度
    [3]d_model = 512  # Embedding Size  词向量大小
    [4]d_ff = 2048  # FeedForward dimension 
    [5]d_k = d_v = 64  # dimension of K(=Q), V   K，Q，V向量长度
    [6]n_layers = 6  # number of Encoder of Decoder Layer   encoder和decoder的长度
    [6]n_heads = 8  # number of heads in Multi-Head Attention   mul-head attention的头数
'''

class transformer_translate(Interface.Interface):
    def __init__(self, \
                input_data = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E'],\
                src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}, \
                tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6},\
                src_len = 5, tgt_len = 5, d_model = 512, d_ff = 2048, d_k = 64, d_v = 64, n_layers = 6, n_heads = 8):
        #super(Interface, self).__init__()
        self.input_data = input_data
        
        self.src_vocab = src_vocab
        self.src_vocab_size = len(src_vocab)
        self.tgt_vocab = tgt_vocab
        self.number_dict = {i: w for i, w in enumerate(tgt_vocab)}
        self.tgt_vocab_size = len(tgt_vocab)

        self.src_len = src_len
        self.tgt_len = tgt_len
        self.d_model = d_model
        self.d_ff = d_ff

        self.d_k = d_k
        self.d_v = d_v
        self.n_layers = n_layers
        self.n_heads = n_heads

    # 控制流程
    def process(self):
        #self.data_process()
        self.enc_inputs, self.dec_inputs, self.target_batch = self.make_batch(self.input_data)
        self.update_parameters()
        self.model()
        self.optimization()
        self.train()
        #self.predict()
        self.test()

    def update_parameters(self):
        self.parameters_name_list = ['src_len', 'tgt_len', 'd_model', 'd_ff','d_k','d_v','n_layers','n_heads']
        self.parameters_list = [self.src_len, self.tgt_len, self.d_model, self.d_ff, self.d_k,self.d_v, self.n_layers, self.n_heads]
        parameters_int_list = ['src_len', 'tgt_len', 'd_model', 'd_ff','d_k','d_v','n_layers','n_heads'] # 输入为int
        parameters_float_list = [] # 输入为float

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
        input_batch = [[self.src_vocab[n] for n in self.input_data[0].split()]]
        output_batch = [[self.tgt_vocab[n] for n in self.input_data[1].split()]]
        target_batch = [[self.tgt_vocab[n] for n in self.input_data[2].split()]]
        return torch.LongTensor(input_batch), torch.LongTensor(output_batch), torch.LongTensor(target_batch)

    def data_process(self):
        # 得到词典，获取词典数目
        self.word_dict, self.number_dict, self.vocab_size = get_dictionary_and_num(self.input_data)


    def model(self):
        self.transformer_model = Transformer(self.src_len , self.tgt_len, self.d_model, self.d_ff, self.d_k, self.d_v, self.n_layers, self.n_heads\
            , self.src_vocab_size, self.tgt_vocab_size)


    def optimization(self):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.transformer_model.parameters(), lr=0.001)

    def train(self):
        print('start train!')
        for epoch in range(20):
            self.optimizer.zero_grad()
            self.outputs, self.enc_self_attns, self.dec_self_attns, self.dec_enc_attns = self.transformer_model(self.enc_inputs, self.dec_inputs)
            loss = self.criterion(self.outputs, self.target_batch.contiguous().view(-1))
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
            loss.backward()
            self.optimizer.step()

    def predict(self):
        pass
    
    def test(self):
        predict, _, _, _ = self.transformer_model(self.enc_inputs, self.dec_inputs)
        print(predict)
        predict = predict.data.max(1, keepdim=True)[1]
        print(self.input_data[0], '->', [self.number_dict[n.item()] for n in predict.squeeze()])