import torch
import torch.nn as nn
import torch.optim as optim
import numpy
from Interface import Interface
from Logistic.dataprocess.bert_dataprocess import bert_dataprocess
from Logistic.model.Bert import Bert

'''
功能：
    使用Bert模型，进行完形填空以及预测下一句任务
输入（可调整的参数）：
    [1]INPUT_DIM = len(en2id)
    [2]OUTPUT_DIM = len(ch2id)
    # 超参数
    [3]BATCH_SIZE = 32
    [4]ENC_EMB_DIM = 256
    [5]DEC_EMB_DIM = 256
    [6]HID_DIM = 512
    [7]N_LAYERS = 2
    [8]ENC_DROPOUT = 0.5
    [9]DEC_DROPOUT = 0.5
    [10LEARNING_RATE = 1e-4
    [11]N_EPOCHS = 20
    [12]CLIP = 1
    [13]bidirectional = True
    [14]attn_method = "general"
    [15]seed = 2020
    [16]input_data = '../datasets/cmn-eng/cmn-1.txt'
输出：
    针对随机抽取的两个句子预测出mask的词语，并判断第二句是否为第一句的后一句
'''



class Bert_premodel_for_NLP(Interface.Interface):
    '''
    maxlen 表示同一个 batch 中的所有句子都由 30 个 token 组成，不够的补 PAD（这里实现的方式比较粗暴，直接固定所有 batch 中的所有句子都为 30）
    max_pred 表示最多需要预测多少个单词，即 BERT 中的完形填空任务
    n_layers 表示 Encoder Layer 的数量
    d_model 表示 Token Embeddings、Segment Embeddings、Position Embeddings 的维度
    d_ff 表示 Encoder Layer 中全连接层的维度
    n_segments 表示 Decoder input 由几句话组成
    '''
    def __init__(self, maxlen = 30,
                       batch_size = 6,
                       max_pred = 5,  # max tokens of prediction
                       n_layers = 6,
                       n_heads = 12,
                       d_model = 768,
                       d_ff = 768 * 4,  # 4*d_model, FeedForward dimension
                       d_k = 64,
                       d_v = 64,  # dimension of K(=Q), V
                       n_segments = 2,
                       input = ('Hello, how are you? I am Romeo.\n'
                                'Hello, Romeo My name is Juliet. Nice to meet you.\n'
                                'Nice meet you too. How are you today?\n'
                                'Great. My baseball team won the competition.\n'
                                'Oh Congratulations, Juliet\n'
                                'Thank you Romeo\n'
                                'Where are you going today?\n'
                                'I am going shopping. What about you?\n'
                                'I am going to visit my grandmother. she is not very well')):
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
        self.input = input

    def process(self):
        print('=' * 80)
        print(self.input)
        print('=' * 80)
        self.data_process()
        self.model()
        self.optimization()
        self.train()
        self.predict()

    def data_process(self):
        self.batch, self.vocab_size, self.word2idx, self.idx2word, self.input_ids,\
        self.segment_ids, self.masked_tokens, self.masked_pos,\
        self.isNext, self.loader = bert_dataprocess(self.input, self.batch_size, self.max_pred, self.maxlen)

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


