# NLP-learning
This is a NLP project.  
Mentor:   
[@Min Gao](http://www.cse.cqu.edu.cn/info/2096/3497.htm)  
Member:   
 [@Yinqiu Huang](https://github.com/964070525); [@Zongwei Wang](https://github.com/CoderWZW); [@Shuai Zhang](https://github.com/1102173230);  [@Jia Wang](https://github.com/JJia000);

Our email:  
Yinqiu Huang yinqiu@cqu.edu.cn  
Zongwei Wang zongwei@cqu.edu.cn  
Shuai Zhang zhangshuai@cqu.edu.cn  
Jia Wang jiawang@cqu.edu.cn  

# 操作说明
1. 进入Main  
2. 执行代码python main.py  
3. 选择所需系统（数据处理系统；模型训练系统（专业推荐，自己得到模型参数）；模型测试系统（非专业推荐, 使用已有模型参数））  
a. 数据处理系统  
b. 模型训练系统  
选择所需模型接口 -> 修改模型参数 -> 训练模型参数及保存   
c. 模型测试系统  
选择数据集 -> 选择训练模型参数 -> 得出所需结果  
更详细的操作说明请见（给链接，带图）

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
