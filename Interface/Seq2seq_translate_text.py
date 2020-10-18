import sys
import time

from Logistic.evaluation.seq2seq_evaluation import evaluate

sys.path.append("..")

import torch
import torch.nn as nn
import torch.optim as optim
import numpy
from Interface import Interface
from Logistic.dataprocess.seq2seq_dataprocess import *
from Logistic.model.Seq2seq import *
from torch.utils.data import Dataset, DataLoader

'''
功能：
    使用Seq2seq模型，将英语句子翻译成中文句子
输入（可调整的参数）：
    [1]INPUT_DIM = len(en2id)    英文词语词汇量
    [2]OUTPUT_DIM = len(ch2id)    中文词语词汇量
    # 超参数
    [3]BATCH_SIZE = 32       Batch大小
    [4]ENC_EMB_DIM = 256     encoder输入维度
    [5]DEC_EMB_DIM = 256     decoder输出维度
    [6]HID_DIM = 512         隐藏层大小
    [7]N_LAYERS = 2          层数
    [8]ENC_DROPOUT = 0.5     encoder中dropout概率
    [9]DEC_DROPOUT = 0.5     decoder中dropout概率
    [10LEARNING_RATE = 1e-4   学习率
    [11]N_EPOCHS = 20        训练次数
    [12]CLIP = 1             梯度裁剪阈值
    [13]bidirectional = True   是否双向
    [14]attn_method = "general"    attention中采取的计算相关性的方法
    [15]seed = 2020        随机种子，这里用于每次做测试的是一样的句子
    [16]input_data = '../datasets/cmn-eng/cmn-1.txt'    训练数据集
输出：
    针对随机抽取的十条英语句子翻译成中文句子
'''

class Seq2seq_translate_text(Interface.Interface):
    def __init__(self, BATCH_SIZE = 32,
                       ENC_EMB_DIM = 256,
                       DEC_EMB_DIM = 256,
                       HID_DIM = 512,
                       N_LAYERS = 2,
                       ENC_DROPOUT = 0.5,
                       DEC_DROPOUT = 0.5,
                       LEARNING_RATE = 1e-4,
                       N_EPOCHS = 20,
                       CLIP = 1,
                       bidirectional = True,
                       attn_method = "general",
                       seed = 2020,
                       input_data='../datasets/cmn-eng/cmn-1.txt'):
        self.BATCH_SIZE = BATCH_SIZE
        self.ENC_EMB_DIM = ENC_EMB_DIM
        self.DEC_EMB_DIM = DEC_EMB_DIM
        self.HID_DIM = HID_DIM
        self.N_LAYERS = N_LAYERS
        self.ENC_DROPOUT = ENC_DROPOUT
        self.DEC_DROPOUT = DEC_DROPOUT
        self.LEARNING_RATE = LEARNING_RATE
        self.N_EPOCHS = N_EPOCHS
        self.CLIP = CLIP
        self.bidirectional = bidirectional
        self.attn_method = attn_method
        self.seed = seed
        self.input_data = input_data

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.basic_dict = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}

    def process(self):
        self.data_process()
        self.model()
        self.optimization()
        self.train()
        self.predict()

    def data_process(self):
        self.en2id, self.id2en, self.ch2id, self.id2ch, self.en_num_data, self.ch_num_data, self.train_set = \
                 seq2seq_dataprocess(self.input_data)
        self.train_loader = DataLoader(self.train_set, batch_size=self.BATCH_SIZE, collate_fn=self.padding_batch)

    def model(self):
        self.INPUT_DIM = len(self.en2id)
        self.OUTPUT_DIM = len(self.ch2id)
        enc = Encoder(self.INPUT_DIM, self.ENC_EMB_DIM, self.HID_DIM, self.N_LAYERS, self.ENC_DROPOUT, self.bidirectional)
        dec = AttnDecoder(self.OUTPUT_DIM, self.DEC_EMB_DIM, self.HID_DIM, self.N_LAYERS, self.DEC_DROPOUT, self.bidirectional, self.attn_method)
        self.model = Seq2Seq(enc, dec, self.device, basic_dict=self.basic_dict).to(self.device)

    def optimization(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE)

    def evaluation(self):
        pass

    def train(self):
        print('start train!')

        self.best_valid_loss = float('inf')
        for epoch in range(self.N_EPOCHS):

            self.start_time = time.time()
            self.train_loss = self.subtrain(self.model, self.train_loader, self.optimizer, self.CLIP)
            self.valid_loss = evaluate(self.model, self.train_loader)
            self.end_time = time.time()

            if self.valid_loss < self.best_valid_loss:
                self.best_valid_loss = self.valid_loss
                torch.save(self.model.state_dict(), "../model/en2ch-attn-model.pt")

            if epoch % 2 == 0:
                epoch_mins, epoch_secs = self.epoch_time(self.start_time, self.end_time)
                print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
                print(f'\tTrain Loss: {self.train_loss:.3f} | Val. Loss: {self.valid_loss:.3f}')

        print("best valid loss：", self.best_valid_loss)


    def predict(self):
        print("start predict!")
        # 加载最优权重
        self.model.load_state_dict(torch.load("../model/en2ch-attn-model.pt"))

        random.seed(self.seed)
        for i in random.sample(range(len(self.en_num_data)), 10):  # 随机看10个
            en_tokens = list(filter(lambda x: x != 0, self.en_num_data[i]))  # 过滤零
            ch_tokens = list(filter(lambda x: x != 3 and x != 0, self.ch_num_data[i]))  # 原文、和机器翻译作对照
            sentence = [self.id2en[t] for t in en_tokens]
            print("【原文】")
            print("".join(sentence))
            translation = [self.id2ch[t] for t in ch_tokens]
            print("【原文】")
            print("".join(translation))
            test_sample = {}
            test_sample["src"] = torch.tensor(en_tokens, dtype=torch.long, device=self.device).reshape(-1, 1)
            test_sample["src_len"] = [len(en_tokens)]
            print("【机器翻译】")
            print(self.translate(self.model, test_sample, self.id2ch), end="\n\n")

    def test(self):
        pass

    # 训练过程
    def subtrain(self,
            model,
            data_loader,
            optimizer,
            clip=1,
            teacher_forcing_ratio=0.5,
            print_every=None  # None不打印
    ):
        model.predict = False
        model.train()

        if print_every == 0:
            print_every = 1

        print_loss_total = 0  # 每次打印都重置
        start = time.time()
        epoch_loss = 0
        for i, batch in enumerate(data_loader):

            # shape = [seq_len, batch]
            input_batchs = batch["src"]
            target_batchs = batch["trg"]
            # list
            input_lens = batch["src_len"]
            target_lens = batch["trg_len"]

            optimizer.zero_grad()  # 将模型的参数梯度初始化为0

            loss = model(input_batchs, input_lens, target_batchs, target_lens, teacher_forcing_ratio)
            print_loss_total += loss.item()
            epoch_loss += loss.item()
            loss.backward()

            # 梯度裁剪，避免梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()

            if print_every and (i + 1) % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('\tCurrent Loss: %.4f' % print_loss_avg)

        return epoch_loss / len(data_loader)

    # 计算所用时间的函数
    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    # 用来padding的函数
    def padding_batch(self, batch):
        """
        输入: -> dict的列表
            [{'src': [1, 2, 3], 'trg': [1, 2, 3]}, {'src': [1, 2, 2, 3], 'trg': [1, 2, 2, 3]}]
        输出: -> 补全了的，张量的dict
            {
                "src": [[1, 2, 3, 0], [1, 2, 2, 3]].T
                "trg": [[1, 2, 3, 0], [1, 2, 2, 3]].T
            }
        """
        # 保存原始句子长度
        src_lens = [d["src_len"] for d in batch]
        trg_lens = [d["trg_len"] for d in batch]

        # 保存原始句子长度中的最大值
        src_max = max([d["src_len"] for d in batch])
        trg_max = max([d["trg_len"] for d in batch])

        # 对于长度不足max的进行<pad>填充
        for d in batch:
            d["src"].extend([self.en2id["<pad>"]] * (src_max - d["src_len"]))
            d["trg"].extend([self.ch2id["<pad>"]] * (trg_max - d["trg_len"]))
        # 整合成张量形式
        srcs = torch.tensor([pair["src"] for pair in batch], dtype=torch.long, device=self.device)
        trgs = torch.tensor([pair["trg"] for pair in batch], dtype=torch.long, device=self.device)

        batch = {"src": srcs.T, "src_len": src_lens, "trg": trgs.T, "trg_len": trg_lens}
        return batch

    # 使用模型进行翻译
    def translate(self,
            model,
            sample,
            idx2token=None
    ):
        model.predict = True
        model.eval()

        # shape = [seq_len, 1]
        input_batch = sample["src"]
        # list
        input_len = sample["src_len"]

        output_tokens = model(input_batch, input_len)
        output_tokens = [idx2token[t] for t in output_tokens]

        return "".join(output_tokens)