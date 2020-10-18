# NLP-learning
record the Input and Output of Interfaces

# NNLM_predict_N_word
功能：
    使用NNLM模型，利用前N-1个word预测第N个word  
输入（可调整的参数）：  
    [1]词库大小(n_class)     
    [2]转化的词向量大小(m)        
    [3]输入层神经元数(即词的滑动窗口容量, n_step)            
    [4]隐层神经元数量(n_hidden)            
    [5]输出层神经元数(n_class)
输出：
    针对给出的前N-1个word预测出的第N个word的list序列

# textCNN_classify
功能：  
    使用textCNN模型进行文本分类
输入（可调整的参数）：  
    [1]embedding_size = 2 # embedding size  词向量大小
    [2]sequence_length = 6 # sequence length  文本长度
    [3]num_classes = 2 # number of classes  标签类别数量
    [4]filter_sizes = [2, 3, 4] # n-gram windows    卷积核的高度
    [5]num_filters = 2 # number of filters  卷积核的组数

# transformer_translate
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
    
# FastText_classify_text
功能：
    使用FastText模型，将一段文本进行快速分类
输入：
    [1]intput: 数据文件，其中每一行为  
           标签 + 中铁 物资 最快 今年底 A H股 上市 募资 120 亿 陈 姗姗 募集 资金 100 亿到 120 亿元 央企 中国......
    [2]lr: 学习率       
    [3]dim： 词向量维度          
    [4]epoch： 训练次数
    [5]word_ngram:  做ngram的窗口大小
    [6]loss:  损失函数类型
    
# Seq2seq_translate_text
功能：
    使用Seq2seq模型，将英语句子翻译成中文句子
输入（可调整的参数）：
    [1]INPUT_DIM = len(en2id)    英文词语词汇量
    [2]OUTPUT_DIM = len(ch2id)    中文词语词汇量
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
    
# Bert_premodel_for_NLP
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
