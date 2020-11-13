import os
import sys

from sklearn.model_selection import train_test_split

sys.path.append("..")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
from tqdm import tqdm
import collections
import random
import time
import torchtext.vocab as Vocab
from Interface import Interface
# from Logistic.dataprocess.get_dictionary_and_num import *
from Logistic.dataprocess.get_tokenized import *
from Logistic.model.textRNN import TextRNN
from Logistic.dataprocess.Data_process import *
from Logistic.dataprocess.get_dictionary_and_num import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
功能：  
    使用textRNN模型进行文本分类
输入（可调整的参数）：  
    [1]embedding_size = 100 # embedding size  词向量大小
    [2]sequence_length = 500 # sequence length  文本长度
    [3]num_classes = 2 # number of classes  标签类别数量
    [4]num_layers = 2 # number of layers    双向LSTM的层数
    [5]num_hiddens = 100 # number of hiddens  隐藏层的大小
    [6]just_test = False # 是否只用于预测
'''

MODEL_ROOT = "../Data/model/textRNN_classify/"
DATA_ROOT = "../Data/train_data/"

class textRNN_classify(Interface.Interface):
    def __init__(self, filename, embedding_size=100, sequence_length=500, num_classes=2, num_layers=2, num_hiddens = 100, just_test=False):
        #super(Interface, self).__init__()
        self.input_data = []
        self.input_labels = []
        self.vocab_size = 0
        self.filename = filename
        self.embedding_size = embedding_size
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_hiddens = num_hiddens
        self.learning_rate = 0.01
        self.num_epochs = 5
        self.batch_size = 64
        self.just_test = just_test

    # 控制流程
    def process(self):
        print("loading data...")
        self.data_process()
        print("loading data succeed!")
        self.update_parameters()
        self.make_batch()
        self.model()
        if self.just_test == False:
            self.optimization()
            self.train()
        #self.predict()
        # self.test()
    
    def update_parameters(self):
        self.parameters_name_list = ['embedding_size', 'sequence_length', 'num_classes', 'num_layers','num_hiddens']
        self.parameters_list = [self.embedding_size, self.sequence_length, self.num_classes, self.num_layers, self.num_hiddens]
        parameters_int_list = ['embedding_size', 'sequence_length', 'num_classes', 'num_layers','num_hiddens'] # 输入为int
        parameters_float_list = [] # 输入为float
        parameters_list_list = [] # 输入为list

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
                        elif self.parameters_name_list[input_number] in parameters_list_list:
                            parameter = list(str(parameter).split(','))
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

    def make_batch(self):
        train_set = Data.TensorDataset(*self.train_data)
        test_set = Data.TensorDataset(*self.test_data)
        self.train_iter = Data.DataLoader(train_set, self.batch_size, shuffle=True)
        self.test_iter = Data.DataLoader(test_set, self.batch_size)

    def data_process(self):
        # 得到词典
        # self.word_dict, self.number_dict, self.vocab_size = get_dictionary_and_num(self.input_data)
        def read_data(data_root):
            data = read_file(data_root)
            train_data = []
            test_data =[]
            #for temp in raw_data:
            #    data.append([temp[0]+temp[2:]])
            Train_data = np.random.choice(data, int(len(data)*0.7), replace = False)
            Test_data = [i for i in data if i not in Train_data]
            for temp in Train_data:
                train_data.append([temp[2:], int(temp[0])])
            for temp in Test_data:
                test_data.append([temp[2:], int(temp[0])])
            return train_data, test_data

        train_data, test_data = read_data(DATA_ROOT+ self.filename)

        # 分词
        tokenized_data = get_tokenized(train_data)

        def get_vocab_imdb(data):
            counter = collections.Counter([tk for st in tokenized_data for tk in st])
            return Vocab.Vocab(counter, min_freq=5)

        self.vocab = get_vocab_imdb(tokenized_data)
        #dict_save(self.vocab, MODEL_ROOT + "textRNN_token2idx.txt")
        self.vocab_size = len(self.vocab)

        def preprocess_imdb(data, vocab, max_l):
            # 将每条评论通过截断或者补0，使得长度变成500

            def pad(x):
                return x[:max_l] if len(x) > max_l else x + [0] * (max_l - len(x))

            tokenized_data = get_tokenized(data)
            features = torch.tensor([pad([vocab.stoi[word] for word in words]) for words in tokenized_data])
            labels = torch.tensor([score for _, score in data])
            return features, labels

        self.train_data = preprocess_imdb(train_data, self.vocab, self.sequence_length)
        self.test_data = preprocess_imdb(test_data, self.vocab, self.sequence_length)

    def model(self):
        self.textRNN_model = TextRNN(self.embedding_size, self.num_classes, self.num_layers, self.num_hiddens, self.vocab_size)

        glove_vocab = Vocab.GloVe(name='6B', dim=100, cache=os.path.join(DATA_ROOT, "glove"))
        def load_pretrained_embedding(words, pretrained_vocab):
            """从预训练好的vocab中提取出words对应的词向量"""
            embed = torch.zeros(len(words), pretrained_vocab.vectors[0].shape[0]) # 初始化为0
            oov_count = 0 # out of vocabulary
            for i, word in enumerate(words):
                try:
                    idx = pretrained_vocab.stoi[word]
                    embed[i, :] = pretrained_vocab.vectors[idx]
                except KeyError:
                    oov_count += 1
            if oov_count > 0:
                print("There are %d oov words." % oov_count)
            return embed

        self.textRNN_model.embedding.weight.data.copy_(
            load_pretrained_embedding(self.vocab.itos, glove_vocab))
        self.textRNN_model.embedding.weight.requires_grad = False # 直接加载预训练好的, 所以不需要更新它

    def optimization(self):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.textRNN_model.parameters()), lr=self.learning_rate)

    def train(self):
        print('start train!')

        self.textRNN_model = self.textRNN_model.to(device)
        print("training on ", device)
        batch_count = 0
        for epoch in range(self.num_epochs):
            train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
            for X, y in self.train_iter:
                X = X.to(device)
                y = y.to(device)
                y_hat = self.textRNN_model(X)
                l = self.criterion(y_hat, y) 
                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
                train_l_sum += l.cpu().item()
                train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
                n += y.shape[0]
                batch_count += 1
            test_acc = self.evaluate_accuracy(self.test_iter, self.textRNN_model)
            print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
                % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

        torch.save(self.textRNN_model, DATA_ROOT+'/model/textRNN_model.pkl')
        print("The model has been saved in " + MODEL_ROOT[3:] + 'textRNN_model.pkl')

    def evaluate_accuracy(self, data_iter, net, device=None):
        if device is None and isinstance(net, torch.nn.Module):
            # 如果没指定device就使用net的device
            device = list(net.parameters())[0].device 
        acc_sum, n = 0.0, 0
        with torch.no_grad():
            for X, y in data_iter:
                if isinstance(net, torch.nn.Module):
                    net.eval() 
                    acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                    net.train() 
                else:
                    if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                        # 将is_training设置成False
                        acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item() 
                    else:
                        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item() 
                n += y.shape[0]
        return acc_sum / n

    def predict(self):
        pass
    
    def test(self):
        if self.just_test == True:
            self.textRNN_model.load_state_dict(torch.load(DATA_ROOT+'/model/textRNN_model.pkl'))
        def predict_sentiment(net, vocab, sentence):
            """sentence是词语的列表"""
            device = list(net.parameters())[0].device
            sentence = torch.tensor([vocab.stoi[word] for word in sentence], device=device)
            label = torch.argmax(net(sentence.view((1, -1))), dim=1)
            return 'positive' if label.item() == 1 else 'negative'

        print(predict_sentiment(self.textRNN_model, self.vocab, ['this', 'movie', 'is', 'so', 'great'])) # positive
        print(predict_sentiment(self.textRNN_model, self.vocab, ['this', 'movie', 'is', 'so', 'bad'])) # negative