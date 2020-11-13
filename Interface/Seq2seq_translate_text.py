import os
import sys
import time

from Logistic.dataprocess.Data_process import dict_save
from Logistic.evaluation.seq2seq_evaluation import evaluate

sys.path.append("..")

import torch
import torch.nn as nn
import torch.optim as optim
import numpy
from Interface import Interface
#from Logistic.dataprocess.seq2seq_dataprocess import *
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

MODEL_ROOT = "../Data/model/Seq2seq_translate_text/"
DATA_ROOT = "../Data/train_data/"

# 返回分好的{源数据、源数据长度、目标数据、目标数据长度}
class TranslationDataset(Dataset):
    def __init__(self, src_data, trg_data):
        self.src_data = src_data
        self.trg_data = trg_data

        assert len(src_data) == len(trg_data), \
            "numbers of src_data  and trg_data must be equal!"    # 表达式为false时执行

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        src_sample = self.src_data[idx]
        src_len = len(self.src_data[idx])
        trg_sample = self.trg_data[idx]
        trg_len = len(self.trg_data[idx])
        return {"src": src_sample, "src_len": src_len, "trg": trg_sample, "trg_len": trg_len}

class Seq2seq_translate_text(Interface.Interface):
    def __init__(self, filename,
                       BATCH_SIZE = 32,
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
                       seed = 2020):
        self.filename = filename
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

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.basic_dict = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}

    def process(self):
        print("loading data...")
        self.data_process()
        print("loading data succeed!")
        self.make_batch()
        self.model()
        self.optimization()
        self.train()
        self.predict()

    def update_parameters(self):
        self.parameters_name_list = ['BATCH_SIZE', 'ENC_EMB_DIM', 'DEC_EMB_DIM', 'HID_DIM','N_LAYERS','ENC_DROPOUT','DEC_DROPOUT','LEARNING_RATE','N_EPOCHS','CLIP']
        self.parameters_list = [self.BATCH_SIZE, self.ENC_EMB_DIM, self.DEC_EMB_DIM, self.HID_DIM, self.N_LAYERS, self.ENC_DROPOUT,self.DEC_DROPOUT,self.LEARNING_RATE,self.N_EPOCHS,self.CLIP]
        parameters_int_list = ['BATCH_SIZE', 'ENC_EMB_DIM', 'DEC_EMB_DIM', 'HID_DIM','N_LAYERS','N_EPOCHS','CLIP'] # 输入为int
        parameters_float_list = ['ENC_DROPOUT','DEC_DROPOUT','LEARNING_RATE'] # 输入为float
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
            
    def data_process(self):
        def seq2seq_dataprocess(input_data):
            with open(input_data, 'r', encoding='utf-8') as f:
                data = f.read()
            data = data.strip()  # 移除头尾空格或换行符
            data = data.split('\n')

            print('samples number:\n', len(data))
            '''
            >>>print('样本示例:\n', data[0])
            样本数:
              2000
            样本示例:
              Hi.	嗨。	CC-BY 2.0 (France) Attribution: tatoeba.org #538123 (CM) & #891077 (Martha)
            '''

            # 分割英文数据和中文数据，英文数据和中文数据间使用\t也就是tab隔开
            en_data = [line.split('\t')[0] for line in data]
            ch_data = [line.split('\t')[1] for line in data]
            '''
            >>>print('\n英文数据:\n', en_data[:10])
            >>>print('中文数据:\n', ch_data[:10])
            英文数据:
             ['Hi.', 'Hi.', 'Run.', 'Wait!', 'Wait!', 'Hello!', 'I won!', 'Oh no!', 'Cheers!', 'Got it?']
            中文数据:
             ['嗨。', '你好。', '你用跑的。', '等等！', '等一下！', '你好。', '我赢了。', '不会吧。', '乾杯!', '你懂了吗？']
            '''

            # 按字符级切割，并添加<eos>
            en_token_list = [[char for char in line] + ["<eos>"] for line in en_data]
            ch_token_list = [[char for char in line] + ["<eos>"] for line in ch_data]
            '''
            >>>print('\n英文数据:\n', en_token_list[:3])
            >>>print('中文数据:\n', ch_token_list[:3])
            英文数据:
              [['H', 'i', '.', '<eos>'], ['H', 'i', '.', '<eos>'], ['R', 'u', 'n', '.', '<eos>']
            中文数据:
              [['嗨', '。', '<eos>'], ['你', '好', '。', '<eos>'], ['你', '用', '跑', '的', '。', '<eos>'],
            '''

            # 基本字典
            basic_dict = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
            # 分别生成中英文字符级字典
            en_vocab = set(''.join(en_data))  # 把所有英文句子串联起来，得到[HiHiRun.....],然后去重[HiRun....]
            en2id = {char: i + len(basic_dict) for i, char in enumerate(en_vocab)}  # enumerate 遍历函数
            en2id.update(basic_dict)  # 将basic字典添加到英文字典
            id2en = {v: k for k, v in en2id.items()}

            # 分别生成中英文字典
            ch_vocab = set(''.join(ch_data))
            ch2id = {char: i + len(basic_dict) for i, char in enumerate(ch_vocab)}
            ch2id.update(basic_dict)
            id2ch = {v: k for k, v in ch2id.items()}

            # 利用字典，映射数据--字符级别映射（例如Hi.<eos>-->[47, 4, 32, 3]
            en_num_data = [[en2id[en] for en in line] for line in en_token_list]
            ch_num_data = [[ch2id[ch] for ch in line] for line in ch_token_list]
            '''
            >>>print('\nchar:', en_data[3])
            >>>print('index:', en_num_data[3])
            char: Wait!
            index: [15, 39, 11, 34, 46, 3]
            '''

            # 得到处理好的训练数据，包含源英文，源英文长度，目标中文和目标中文长度
            train_set = TranslationDataset(en_num_data, ch_num_data)

            return en2id, id2en, ch2id, id2ch, en_num_data, ch_num_data, train_set
        self.en2id, self.id2en, self.ch2id, self.id2ch, self.en_num_data, self.ch_num_data, self.train_set = \
                 seq2seq_dataprocess(DATA_ROOT+self.filename)

        if os.path.exists(MODEL_ROOT) == False:
            os.mkdir(MODEL_ROOT)
        dict_save(self.en2id, MODEL_ROOT + "Seq2seq_translate_text_en2id.txt")
        dict_save(self.id2en, MODEL_ROOT + "Seq2seq_translate_text_id2en.txt")
        dict_save(self.ch2id, MODEL_ROOT + "Seq2seq_translate_text_ch2id.txt")
        dict_save(self.id2ch, MODEL_ROOT + "Seq2seq_translate_text_id2ch.txt")


    def make_batch(self):
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
                torch.save(self.model, MODEL_ROOT + 'Seq2seq_translate_text.pkl')


            if epoch % 2 == 0:
                epoch_mins, epoch_secs = self.epoch_time(self.start_time, self.end_time)
                print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
                print(f'\tTrain Loss: {self.train_loss:.3f} | Val. Loss: {self.valid_loss:.3f}')
        print("The model has been saved in " + MODEL_ROOT[3:] + 'Seq2seq_translate_text.pkl')
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