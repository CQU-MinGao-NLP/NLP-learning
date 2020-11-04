import sys
sys.path.append("..") 

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Interface import Interface
from Logistic.dataprocess.get_dictionary_and_num import *
from Logistic.model.textCNN import TextCNN
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

class textCNN_classify(Interface.Interface):
    def __init__(self, input_data = ["i love you and like you", "he loves me and like you", "she likes baseball and like you", "i hate you and hate he", "sorry for that i hate you", "this is awful and hate you"],\
                 input_labels = [1, 1, 1, 0, 0, 0], embedding_size = 2, sequence_length = 6, num_classes = 2, filter_sizes = [2,3,4], num_filters = 2):
        #super(Interface, self).__init__()
        self.input_data = input_data
        self.input_labels = input_labels
        self.embedding_size = embedding_size
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters

    # 控制流程
    def process(self):
        self.data_process()
        self.update_parameters()
        #self.make_batch(self.input_data)
        self.model()
        self.optimization()
        self.train()
        #self.predict()
        self.test()
    
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
        self.word_dict, self.number_dict, self.vocab_size = get_dictionary_and_num(self.input_data)


    def model(self):
        self.textCNN_model = TextCNN(self.embedding_size, self.sequence_length, self.num_classes, self.filter_sizes, self.num_filters, self.vocab_size)


    def optimization(self):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.textCNN_model.parameters(), lr=0.001)

    def train(self):
        print('start train!')
        self.tensor_input_batch = torch.LongTensor([np.asarray([self.word_dict[n] for n in sen.split()]) for sen in self.input_data])
        self.tensor_target_batch = torch.LongTensor([out for out in self.input_labels]) # To using Torch Softmax Loss function

        # Training
        for epoch in range(5000):
            self.optimizer.zero_grad()
            output = self.textCNN_model(self.tensor_input_batch)

            # output : [batch_size, n_class], target_batch : [batch_size]
            loss = self.criterion(output, self.tensor_target_batch)
            if (epoch + 1) % 1000 == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

            loss.backward()
            self.optimizer.step()

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