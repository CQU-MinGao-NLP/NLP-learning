# NLP-learning
This is a NLP project.
Mentor: Min Gao
Member: [@Shuai Zhang](https://github.com/1102173230); [@Wangzongwei](https://github.com/CoderWZW); [@Wangjia](https://github.com/JJia000);

our email:  
ZhuangShuai zhangshuai@cqu.edu.cn  
Wangzongwei zongwei@cqu.edu.cn  
wangjia jjia@cqu.edu.cn

# 代码添加逻辑
下面以NNLM预测第N个word功能为例：
1. 在Data中添加数据处理代码  
因为在整个过程用的数据很少，所以这里没添加
2. 在Logistic添加模块功能
a. Logistic.model中增加NNLM.py，增加模型结构的代码。
b. Logistic.dataprocess中增加get_dictionary_and_num.py，增加数据加工的代码。
3. 在Interface里面定义接口逻辑  
a. 在NNLM_predict_next_word.py编写逻辑，请继承Interface这个接口，然后使用process来管理进程逻辑。  
b. 在Readme中添加接口信息  
4. 在Main中增加choose选项

注意：因为example比较简单，有些板块没有涉及到，可以自己尝试加在其他模块，如果有些地方不知道怎么添加在飞书中联系。

# 操作说明
进入Main, 执行代码  
python main.py
