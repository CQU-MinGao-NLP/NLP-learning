import sys
import os
sys.path.append("..") 

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Interface import Interface
from Logistic.dataprocess.get_dictionary_and_num import *
from Logistic.dataprocess.Data_process import dict_save
from Logistic.model.transformer import Transformer
from torch.utils.data import Dataset, DataLoader
from Logistic.dataprocess.Data_process import *
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

MODEL_ROOT = "../Data/model/transformer_translate/"
DATA_ROOT = "../Data/train_data/"

class transformer_translate(Interface.Interface):
    def __init__(self, \
                # input_data = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E'],\
                # src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}, \
                # tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6},\
                filename,
                src_len = 10, tgt_len = 10, d_model = 512, d_ff = 2048, d_k = 64, d_v = 64, n_layers = 6, n_heads = 8):
        #super(Interface, self).__init__()
        # self.input_data = input_data

        self.filename = filename
        # self.src_vocab = src_vocab
        # self.src_vocab_size = len(src_vocab)
        # self.tgt_vocab = tgt_vocab
        # self.number_dict = {i: w for i, w in enumerate(tgt_vocab)}
        # self.tgt_vocab_size = len(tgt_vocab)

        self.src_len = src_len
        self.tgt_len = tgt_len
        self.d_model = d_model
        self.d_ff = d_ff

        self.d_k = d_k
        self.d_v = d_v
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.batch_size = 3
        self.num_epochs = 2
    # 控制流程
    def process(self):
        print("loading data...")
        self.data_process()
        print("loading data succeed!")
        self.make_batch()
        self.update_parameters()
        self.model()
        self.optimization()
        self.train()
        #self.predict()
        #self.test()

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

    def data_process(self):

        with open(DATA_ROOT+self.filename, 'r', encoding='utf-8') as f:
            data = f.read()
        data = data.strip()  # 移除头尾空格或换行符
        data = data.split('\n')

        print('samples number:\n', len(data))

        # 分割源数据和目标数据，源数据和目标数据间使用\t也就是tab隔开(string)
        input_data = [line.split('\t')[0] for line in data]
        output_data = [line.split('\t')[1] for line in data]
        target_data = [line.split('\t')[1] for line in data]

        def pad(x, max_l):
            return x[:max_l] if len(x) > max_l else x + ["<pad>"] * (max_l - len(x))

        # 按词级切割，并添加<eos> (word)
        input_corpus = [en_tokenizer([line])[0] for line in input_data]
        output_corpus = [cn_tokenizer([line])[0] for line in output_data]
        target_corpus = [cn_tokenizer([line])[0] for line in target_data]

        # 基本字典
        basic_dict = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
        # 分别生成中英文字符级字典
        
        en_vocab = set(word for sen in input_corpus for word in sen)  # 把所有英文句子串联起来，得到[HiHiRun.....],然后去重[HiRun....]
        
        en2id = {char: i + len(basic_dict) for i, char in enumerate(en_vocab)}  # enumerate 遍历函数
        en2id.update(basic_dict)  # 将basic字典添加到英文字典
        id2en = {v: k for k, v in en2id.items()}

        # 分别生成中英文字典
        ch_vocab = set(word for sen in target_corpus for word in sen)
        # print('len_chvocab', len(list(ch_vocab)))
        ch2id = {char: i + len(basic_dict) for i, char in enumerate(ch_vocab)}
        ch2id.update(basic_dict)
        id2ch = {v: k for k, v in ch2id.items()}

        if os.path.exists(MODEL_ROOT) == False: 
            os.mkdir(MODEL_ROOT)
        dict_save(en2id, MODEL_ROOT + "transformer_translate_en2id.txt")
        dict_save(id2en, MODEL_ROOT + "transformer_translate_id2en.txt")
        dict_save(ch2id, MODEL_ROOT + "transformer_translate_ch2id.txt")
        dict_save(id2ch, MODEL_ROOT + "transformer_translate_id2ch.txt")

        input_list = [pad(en_tokenizer([line])[0]  + ["<pad>"], self.src_len) for line in input_data]
        output_list = [["<bos>"] + pad(cn_tokenizer([line])[0], self.tgt_len - 1) for line in output_data]
        target_list = [pad(cn_tokenizer([line])[0] + ["<eos>"], self.tgt_len) for line in target_data]
        # 利用字典，映射数据--字符级别映射（例如Hi.<eos>-->[47, 4, 32, 3]
        en_num_data = [[en2id[en] for en in line] for line in input_list]
        ch_num_data_dec_input = [[ch2id[ch] for ch in line] for line in output_list]
        ch_num_data_dec_output = [[ch2id[ch] for ch in line] for line in target_list]

        self.train_set = torch.utils.data.TensorDataset(torch.tensor(en_num_data), torch.tensor(ch_num_data_dec_input), torch.tensor(ch_num_data_dec_output))
        self.src_vocab_size = len(en2id)
        self.tgt_vocab_size = len(ch2id)
        # print('len(en2id)', len(en2id))
        # print('len(ch2id)', len(ch2id))


    def make_batch(self):
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size)

    def model(self):
        self.transformer_model = Transformer(self.src_len, self.tgt_len, self.d_model, self.d_ff, self.d_k, self.d_v, self.n_layers, self.n_heads\
            , self.src_vocab_size, self.tgt_vocab_size)


    def optimization(self):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.transformer_model.parameters(), lr=0.001)

    def train(self):
        print('start train!')
        for epoch in range(self.num_epochs):
            self.optimizer.zero_grad()
            for enc_inputs, dec_inputs, dec_output in self.train_loader:
                # print(enc_inputs)
                # print(dec_inputs)
                # print(dec_output)
                # print(self.src_vocab_size)
                # print(self.tgt_vocab_size)
                self.outputs, self.enc_self_attns, self.dec_self_attns, self.dec_enc_attns = self.transformer_model(enc_inputs, dec_inputs)
                loss = self.criterion(self.outputs, dec_output.contiguous().view(-1))
                loss.backward()
                self.optimizer.step()
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        if os.path.exists(MODEL_ROOT) == False:
            os.mkdir(MODEL_ROOT)
        torch.save(self.transformer_model, MODEL_ROOT + 'transformer_translate.pkl')
        print("The model has been saved in " + MODEL_ROOT[3:] + 'transformer_translate.pkl')

    def predict(self):
        pass
    
    def test(self):
        predict, _, _, _ = self.transformer_model(self.enc_inputs, self.dec_inputs)
        print(predict)
        predict = predict.data.max(1, keepdim=True)[1]
        print(self.input_data[0], '->', [self.number_dict[n.item()] for n in predict.squeeze()])