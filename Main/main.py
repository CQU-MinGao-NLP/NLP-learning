import sys
sys.path.append("..") 

import torch
import torch.nn as nn
import torch.optim as optim
import numpy
from Interface.NNLM_predict_next_word import NNLM_predict_N_word

# 开始界面
def start():
    print('*'*80)
    print('Here is Help more people, natural language process (HMP-NLP)')
    print('This system has two ways to operate, we automatically use the system way, if you would like to use it by yourself, please contact us accordding to ReadMe')
    print('Members: Wang zongwei, Zhang Shuai, Wang Jia')
    print('Mentor: Gao Min')
    print('Organization: Chongqing University')

# 选择所需的接口
def choose():
    print('*'*80)
    print('There are some basic models to choose, please choose one')
    print('(example: if you input 1 with <enter>, you will run NNLM)')
    print('1. NNLM_predict_next_word(sample)')
    print('*'*80)
    number = int(input())
    return number

# 结束界面
def end():
    print('*'*80)
    print('It is the ending! If you would like to use it again, please run again~')
    print('*'*80)
if __name__ == '__main__':
    start()
    number = choose()
    if number == 1:
        test = NNLM_predict_N_word()
        test.process()
    else:
        pass
    end()
