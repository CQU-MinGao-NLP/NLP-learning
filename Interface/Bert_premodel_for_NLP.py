import os
import re
from random import randrange, shuffle, random, randint

import torch
import torch.nn as nn
import torch.optim as optim
import numpy
from Interface import Interface
from Logistic.dataprocess.Data_process import read_file
from Logistic.dataprocess.bert_dataprocess import bert_dataprocess
from Logistic.model.Bert import Bert
import torch.utils.data as Data
from Logistic.dataprocess.Data_process import dict_save

'''
功能：
    使用Bert模型，进行完形填空以及预测下一句任务
输入（可调整的参数）：
    [1] maxlen 表示同一个 batch 中的所有句子都由 30 个 token 组成，不够的补 PAD（这里实现的方式比较粗暴，直接固定所有 batch 中的所有句子都为 30）
    [2] batch_size 表示batch大小
    [3] max_pred 表示最多需要预测多少个单词，即 BERT 中的完形填空任务
    [4] n_layers 表示 Encoder Layer 的数量
    [5] n_heads 表示多头注意力机制个数
    [6] d_model 表示 Token Embeddings、Segment Embeddings、Position Embeddings 的维度
    [7] d_ff 表示 Encoder Layer 中全连接层的维度
    [8] d_k 表示K和Q的维度
    [9] d_v 表示V的维度
    [10] n_segments 表示 Decoder input 由几句话组成
    [11] input 表示输入数据集，一段连续文本
输出：
    针对随机抽取的两个句子预测出mask的词语，并判断第二句是否为第一句的后一句
'''

MODEL_ROOT = "../Data/model/Bert_premodel_for_NLP/"
DATA_ROOT = "../Data/train_data/"

class MyDataSet(Data.Dataset):
    def __init__(self, input_ids, segment_ids, masked_tokens, masked_pos, isNext):
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.masked_tokens = masked_tokens
        self.masked_pos = masked_pos
        self.isNext = isNext

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.segment_ids[idx], self.masked_tokens[idx], self.masked_pos[idx], self.isNext[
            idx]

class Bert_premodel_for_NLP(Interface.Interface):
    '''
    maxlen 表示同一个 batch 中的所有句子都由 30 个 token 组成，不够的补 PAD（这里实现的方式比较粗暴，直接固定所有 batch 中的所有句子都为 30）
    max_pred 表示最多需要预测多少个单词，即 BERT 中的完形填空任务
    n_layers 表示 Encoder Layer 的数量
    d_model 表示 Token Embeddings、Segment Embeddings、Position Embeddings 的维度
    d_ff 表示 Encoder Layer 中全连接层的维度
    n_segments 表示 Decoder input 由几句话组成
    '''
    def __init__(self, filename,
                       maxlen = 30,
                       batch_size = 6,
                       max_pred = 5,  # max tokens of prediction
                       n_layers = 6,
                       n_heads = 12,
                       d_model = 768,
                       d_ff = 768 * 4,  # 4*d_model, FeedForward dimension
                       d_k = 64,
                       d_v = 64,  # dimension of K(=Q), V
                       n_segments = 2,
                       ):
        self.maxlen = maxlen
        self.batch_size = batch_size
        self.max_pred = max_pred
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_k = d_k
        self.d_v = d_v
        self.n_segments = n_segments
        self.filename = filename

    def process(self):
        print("loading data...")
        self.data_process()
        print("loading data succeed!")
        self.update_parameters()
        self.make_batch()
        self.model()
        self.optimization()
        self.train()
        # self.predict()

    def update_parameters(self):
        self.parameters_name_list = ['maxlen', 'batch_size', 'max_pred', 'n_layers','n_heads','d_model','d_ff','d_k','d_v','n_segments']
        self.parameters_list = [self.maxlen, self.batch_size, self.max_pred, self.n_layers, self.n_heads,self.d_model, self.d_ff, self.d_k,self.d_v,self.n_segments]
        parameters_int_list = ['maxlen', 'batch_size', 'max_pred', 'n_layers','n_heads','d_model','d_ff','d_k','d_v','n_segments'] # 输入为int
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
        # self.batch, self.vocab_size, self.word2idx, self.idx2word, self.input_ids,\
        # self.segment_ids, self.masked_tokens, self.masked_pos,\
        # self.isNext, self.loader = bert_dataprocess(self.input, self.batch_size, self.max_pred, self.maxlen)
        # self.input = read_file(DATA_ROOT + self.filename)
        with open(DATA_ROOT + self.filename) as f:
            self.input = f.read()

        # 数据处理
        self.sentences = re.sub("[.,!?\\-]", '', self.input.lower()).split('\n')  # filter '.', ',', '?', '!'
        word_list = list(set(" ".join(self.sentences).split()))  # ['hello', 'how', 'are', 'you',...]
        self.word2idx = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
        for i, w in enumerate(word_list):
            self.word2idx[w] = i + 4
        self.idx2word = {i: w for i, w in enumerate(self.word2idx)}
        self.vocab_size = len(self.word2idx)

        if os.path.exists(MODEL_ROOT) == False:
            os.mkdir(MODEL_ROOT)
        dict_save(self.idx2word, MODEL_ROOT + "Bert_idx2token.txt")
        dict_save(self.word2idx, MODEL_ROOT + "Bert_token2idx.txt")

        # 存储全部文本每句话对应的word的index，二维
        self.token_list = list()
        for sentence in self.sentences:
            arr = [self.word2idx[s] for s in sentence.split()]
            self.token_list.append(arr)

    def make_batch(self):
        self.batch = []
        positive = negative = 0
        while positive != self.batch_size / 2 or negative != self.batch_size / 2:

            # 预测下一句
            tokens_a_index, tokens_b_index = randrange(len(self.sentences)), randrange(
                len(self.sentences))  # sample random index in sentences
            tokens_a, tokens_b = self.token_list[tokens_a_index], self.token_list[tokens_b_index]
            self.input_ids = [self.word2idx['[CLS]']] + tokens_a + [self.word2idx['[SEP]']] + tokens_b + [self.word2idx['[SEP]']]
            self.segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)

            # 完形填空任务
            n_pred = min(self.max_pred, max(1, int(len(self.input_ids) * 0.15)))  # 15 % of tokens in one sentence
            cand_maked_pos = [i for i, token in enumerate(self.input_ids)
                              if token != self.word2idx['[CLS]'] and token != self.word2idx['[SEP]']]  # candidate masked position
            shuffle(cand_maked_pos)
            self.masked_tokens, self.masked_pos = [], []
            for pos in cand_maked_pos[:n_pred]:
                self.masked_pos.append(pos)
                self.masked_tokens.append(self.input_ids[pos])
                if random() < 0.8:  # 80%
                    self.input_ids[pos] = self.word2idx['[MASK]']  # make mask
                elif random() > 0.9:  # 10%
                    index = randint(0, self.vocab_size - 1)  # random index in vocabulary
                    while index < 4:  # can't involve 'CLS', 'SEP', 'PAD'
                        index = randint(0, self.vocab_size - 1)
                    self.input_ids[pos] = index  # replace

            # 使用0进行padding
            n_pad = self.maxlen - len(self.input_ids)
            self.input_ids.extend([0] * n_pad)
            self.segment_ids.extend([0] * n_pad)

            # 是为了补齐 mask 的数量，因为不同句子长度，会导致不同数量的单词进行 mask，我们需要保证同一个 batch 中，mask 的数量（必须）是相同的，
            if self.max_pred > n_pred:
                n_pad = self.max_pred - n_pred
                self.masked_tokens.extend([0] * n_pad)
                self.masked_pos.extend([0] * n_pad)

            # tokens_a_index + 1 == tokens_b_index代表b是a的下一句
            if tokens_a_index + 1 == tokens_b_index and positive < self.batch_size / 2:
                self.batch.append([self.input_ids, self.segment_ids, self.masked_tokens, self.masked_pos, True])  # IsNext
                positive += 1
            elif tokens_a_index + 1 != tokens_b_index and negative < self.batch_size / 2:
                self.batch.append([self.input_ids, self.segment_ids, self.masked_tokens, self.masked_pos, False])  # NotNext
                negative += 1

        self.input_ids, self.segment_ids, self.masked_tokens, self.masked_pos, self.isNext = zip(*self.batch)
        self.input_ids, self.segment_ids, self.masked_tokens, self.masked_pos, self.isNext = \
            torch.LongTensor(self.input_ids), torch.LongTensor(self.segment_ids), torch.LongTensor(self.masked_tokens), \
            torch.LongTensor(self.masked_pos), torch.LongTensor(self.isNext)

        self.loader = Data.DataLoader(MyDataSet(self.input_ids, self.segment_ids, self.masked_tokens, self.masked_pos, self.isNext), self.batch_size, True)


    def model(self):
        self.model = Bert(self.d_model, self.n_layers, self.vocab_size, self.maxlen, self.n_segments,
                          self.d_k, self.d_v, self.n_heads, self.d_ff)

    def optimization(self):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adadelta(self.model.parameters(), lr=0.001)

    def evaluation(self):
        pass

    def train(self):
        for epoch in range(50):
            for input_ids, segment_ids, masked_tokens, masked_pos, isNext in self.loader:
                logits_lm, logits_clsf = self.model(input_ids, segment_ids, masked_pos)
                loss_lm = self.criterion(logits_lm.view(-1, self.vocab_size), masked_tokens.view(-1))  # for masked LM
                loss_lm = (loss_lm.float()).mean()
                loss_clsf = self.criterion(logits_clsf, isNext)  # for sentence classification
                loss = loss_lm + loss_clsf
                if (epoch + 1) % 10 == 0:
                    print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        if os.path.exists(MODEL_ROOT) == False:
            os.mkdir(MODEL_ROOT)
        torch.save(self.model, MODEL_ROOT + 'Bert_premodel_for_NLP.pkl')
        print("The model has been saved in " + MODEL_ROOT[3:] + 'Bert_premodel_for_NLP.pkl')

    def predict(self):
        input_ids, segment_ids, masked_tokens, masked_pos, isNext = self.batch[1]
        print([self.idx2word[w] for w in input_ids if self.idx2word[w] != '[PAD]'])

        logits_lm, logits_clsf = self.model(torch.LongTensor([input_ids]), \
                                       torch.LongTensor([segment_ids]), torch.LongTensor([masked_pos]))
        logits_lm = logits_lm.data.max(2)[1][0].data.numpy()
        print('masked tokens list : ', [pos for pos in masked_tokens if pos != 0])
        print('predict masked tokens list : ', [pos for pos in logits_lm if pos != 0])

        logits_clsf = logits_clsf.data.max(1)[1].data.numpy()[0]
        print('isNext : ', True if isNext else False)
        print('predict isNext : ', True if logits_clsf else False)

    def test(self):
        pass


