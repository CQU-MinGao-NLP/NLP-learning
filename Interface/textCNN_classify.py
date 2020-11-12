import sys
import time

from Logistic.dataprocess.get_tokenized import get_tokenized

sys.path.append("..")

import collections
import torchtext.vocab as Vocab
from Logistic.dataprocess.Data_process import *
import torch.utils.data as Data
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Interface import Interface
from Logistic.dataprocess.get_dictionary_and_num import *
from Logistic.model.textCNN import TextCNN

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
功能：  
    使用textCNN模型进行文本分类
输入（可调整的参数）：  
    [1]embedding_size = 2 # embedding size  词向量大小
    [2]sequence_length = 6 # sequence length  文本长度
    [3]num_classes = 2 # number of classes  标签类别数量
    [4]filter_sizes = [2, 3, 4] # n-gram windows    卷积核的高度
    [5]num_filters = 2 # number of filters  卷积核的组数
    [6]vocab_size #number of vocab 词典的大小
'''

MODEL_ROOT = "../Data/model/textCNN_classify/"
DATA_ROOT = "../Data/train_data/"

class textCNN_classify(Interface.Interface):
    def __init__(self, filename,
                 embedding_size = 2, sequence_length = 500, num_classes = 2, filter_sizes = [2,3,4], num_filters = 2):
        #super(Interface, self).__init__()
        self.input_data = []
        self.input_labels = []
        self.filename = filename
        self.embedding_size = embedding_size
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.learning_rate = 0.01
        self.num_epochs = 5
        self.batch_size = 6

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
    
    def update_parameters(self):
        self.parameters_name_list = ['embedding_size', 'sequence_length', 'num_classes', 'filter_sizes','num_filters']
        self.parameters_list = [self.embedding_size, self.sequence_length, self.num_classes, self.filter_sizes, self.num_filters]
        parameters_int_list = ['embedding_size', 'sequence_length', 'num_classes', 'filter_sizes','num_filters'] # 输入为int
        parameters_float_list = [] # 输入为float
        parameters_list_list = ['filter_sizes'] # 输入为list

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

    def data_process(self):
        # 得到词典，获取词典数目
        def read_data(data_root):
            data = read_file(data_root)
            train_data = []
            test_data =[]
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
        self.vocab_size = len(self.vocab)

        def preprocess_imdb(data, vocab):
            max_l = 500  # 将每条评论通过截断或者补0，使得长度变成500

            def pad(x):
                return x[:max_l] if len(x) > max_l else x + [0] * (max_l - len(x))

            tokenized_data = get_tokenized(data)
            features = torch.tensor([pad([vocab.stoi[word] for word in words]) for words in tokenized_data])
            labels = torch.tensor([score for _, score in data])
            return features, labels

        self.train_data = preprocess_imdb(train_data, self.vocab)
        self.test_data = preprocess_imdb(test_data, self.vocab)

        #self.word_dict, self.number_dict, self.vocab_size = get_dictionary_and_num(self.train_data[0])
        #if os.path.exists(MODEL_ROOT) == False:
        #    os.mkdir(MODEL_ROOT)
        #dict_save(self.number_dict, MODEL_ROOT + "textCNN_classify_idx2token.txt")
        #dict_save(self.word_dict, MODEL_ROOT + "textCNN_classify_token2idx.txt")

    def make_batch(self):
        train_set = Data.TensorDataset(*self.train_data)
        test_set = Data.TensorDataset(*self.test_data)
        self.train_iter = Data.DataLoader(train_set, self.batch_size, shuffle=True)
        self.test_iter = Data.DataLoader(test_set, self.batch_size)

    def model(self):
        self.textCNN_model = TextCNN(self.embedding_size, self.sequence_length, self.num_classes, self.filter_sizes, self.num_filters, self.vocab_size)


    def optimization(self):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.textCNN_model.parameters(), lr=0.001)

    # def train(self):
    #     print('start train!')
    #     #self.input_data = self.train_data[0]
    #     #self.input_labels = self.train_data[1]
    #     #self.tensor_input_batch = torch.LongTensor([np.asarray([self.word_dict[n] for n in sen.split()]) for sen in self.input_data])
    #     #self.tensor_target_batch = torch.LongTensor([out for out in self.input_labels]) # To using Torch Softmax Loss function
    #
    #
    #     # Training
    #     for epoch in range(5000):
    #         self.optimizer.zero_grad()
    #         output = self.textCNN_model(self.train_data[0])
    #
    #         # output : [batch_size, n_class], target_batch : [batch_size]
    #         loss = self.criterion(output, self.train_data[1])
    #         if (epoch + 1) % 1000 == 0:
    #             print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
    #
    #         loss.backward()
    #         self.optimizer.step()
    def train(self):
        print('start train!')

        self.textCNN_model = self.textCNN_model.to(device)
        print("training on ", device)
        batch_count = 0
        for epoch in range(self.num_epochs):
            train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
            for X, y in self.train_iter:
                X = X.to(device)
                y = y.to(device)
                y_hat = self.textCNN_model(X)
                l = self.criterion(y_hat, y)
                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
                train_l_sum += l.cpu().item()
                train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
                n += y.shape[0]
                batch_count += 1
            test_acc = self.evaluate_accuracy(self.test_iter, self.textCNN_model)
            print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
                  % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
        torch.save(self.textCNN_model, DATA_ROOT+'/model/textCNN_model.pkl')
        print("The model has been saved in " + MODEL_ROOT[3:] + 'textCNN_model.pkl')

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
        test_text = 'i sorry that i hate you'
        tests = [np.asarray([self.word_dict[n] for n in test_text.split()])]
        test_batch = torch.LongTensor(tests)
        predict = self.textCNN_model(test_batch).data.max(1, keepdim=True)[1]
        if predict[0][0] == 0:
            print(test_text,"is Bad Mean...")
        else:
            print(test_text,"is Good Mean!!")