import re
from random import randrange, shuffle, random, randint

import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as Data
'''
Bert使用的dataprocess
将一段连续文本先进行初始处理：分割word，创建字典；
构建训练所需的数据格式，即从连续文本中随机取两句话拼接，并随机mask15%词汇，如：
    ['[CLS]', 'where', 'are', '[MASK]', 'going', 'today', '[SEP]', 'oh', 'congratulations', 'juliet', '[SEP]']
将构建好的数据格式数据封装并使用数据加载器输出

输入为：
   [1]input:  一整段连续文本：“How are you？I'm fine，thank you！”
   [2]batch_size:  batch大小
   [3]max_pred:  可以mask的最大值
   [4]maxlen:  每句话最长长度，用于padding
           
输出为：
处理好的数据集合：
    batch
字典大小：
    vocab_size
两个字典：
    word2idx和idx2word
处理好的数据格式中的以下信息：
    input_ids：  每行表示每一条文本的index序列
    segment_ids：  每行表示每一条文本的标记符，第一句话统一为0，第二句话统一为1
    masked_tokens：  每行表示每一条文本被mask的word序列
    masked_pos：  每行表示每一条文本被mask的word的index序列
    isNext：  每行表示每一条文本的第二句话是否是第一句的下一句，True or False
一个数据加载器：
    loader
'''

# sample IsNext and NotNext to be same in small batch size
# 数据预处理部分，我们需要根据概率随机mask一句话中 15% 的 token，还需要拼接任意两句话
def bert_dataprocess(input,batch_size,max_pred,maxlen):
    # 数据处理
    sentences = re.sub("[.,!?\\-]", '', input.lower()).split('\n')  # filter '.', ',', '?', '!'
    word_list = list(set(" ".join(sentences).split()))  # ['hello', 'how', 'are', 'you',...]
    word2idx = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
    for i, w in enumerate(word_list):
        word2idx[w] = i + 4
    idx2word = {i: w for i, w in enumerate(word2idx)}
    vocab_size = len(word2idx)

    # 存储全部文本每句话对应的word的index，二维
    token_list = list()
    for sentence in sentences:
        arr = [word2idx[s] for s in sentence.split()]
        token_list.append(arr)


    batch = []
    positive = negative = 0
    while positive != batch_size/2 or negative != batch_size/2:

        # 预测下一句
        tokens_a_index, tokens_b_index = randrange(len(sentences)), randrange(len(sentences)) # sample random index in sentences
        tokens_a, tokens_b = token_list[tokens_a_index], token_list[tokens_b_index]
        input_ids = [word2idx['[CLS]']] + tokens_a + [word2idx['[SEP]']] + tokens_b + [word2idx['[SEP]']]
        segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)

        # 完形填空任务
        n_pred = min(max_pred, max(1, int(len(input_ids) * 0.15))) # 15 % of tokens in one sentence
        cand_maked_pos = [i for i, token in enumerate(input_ids)
                          if token != word2idx['[CLS]'] and token != word2idx['[SEP]']] # candidate masked position
        shuffle(cand_maked_pos)
        masked_tokens, masked_pos = [], []
        for pos in cand_maked_pos[:n_pred]:
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            if random() < 0.8:  # 80%
                input_ids[pos] = word2idx['[MASK]'] # make mask
            elif random() > 0.9:  # 10%
                index = randint(0, vocab_size - 1) # random index in vocabulary
                while index < 4: # can't involve 'CLS', 'SEP', 'PAD'
                  index = randint(0, vocab_size - 1)
                input_ids[pos] = index # replace

        # 使用0进行padding
        n_pad = maxlen - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)

        # 是为了补齐 mask 的数量，因为不同句子长度，会导致不同数量的单词进行 mask，我们需要保证同一个 batch 中，mask 的数量（必须）是相同的，
        if max_pred > n_pred:
            n_pad = max_pred - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)

        # tokens_a_index + 1 == tokens_b_index代表b是a的下一句
        if tokens_a_index + 1 == tokens_b_index and positive < batch_size/2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True]) # IsNext
            positive += 1
        elif tokens_a_index + 1 != tokens_b_index and negative < batch_size/2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False]) # NotNext
            negative += 1

    input_ids, segment_ids, masked_tokens, masked_pos, isNext = zip(*batch)
    input_ids, segment_ids, masked_tokens, masked_pos, isNext = \
        torch.LongTensor(input_ids), torch.LongTensor(segment_ids), torch.LongTensor(masked_tokens), \
        torch.LongTensor(masked_pos), torch.LongTensor(isNext)

    loader = Data.DataLoader(MyDataSet(input_ids, segment_ids, masked_tokens, masked_pos, isNext), batch_size, True)

    return batch, vocab_size, word2idx, idx2word, input_ids, segment_ids, masked_tokens, masked_pos, isNext, loader

# 数据预处理结束

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