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

