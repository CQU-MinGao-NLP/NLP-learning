from torch.utils.data import Dataset, DataLoader
'''
Seq2seq使用的dataprocess
得到字典，获取字典数目
输入格式为 文件地址 其中文件中每一行格式如下：
           “We try. 我们来试试。”
输出为：
四个字典：
    dict-en2id: key是英文char， value是index
    dict-id2en: key是index， value是英文char
    dict-ch2id: key是中文字， value是index
    dict-id2ch: key是index， value是中文字
两个字符级别映射矩阵：
    en_num_data：第i行序列为第i个英语句子对应的每个字符的index值
    ch_num_data：第i行序列为第i个中文句子对应的每个字的index值
一个处理好的训练集：
    train_set:{"src": src_sample, "src_len": src_len, "trg": trg_sample, "trg_len": trg_len}
'''

def seq2seq_dataprocess(input_data):
    with open(input_data, 'r', encoding='utf-8') as f:
        data = f.read()
    data = data.strip()  # 移除头尾空格或换行符
    data = data.split('\n')

    print('样本数:\n', len(data))
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


