import os
import re

'''
Function:
    Determine the validity of various inputs, including [int, filename, yes or no, float]
    It will jump out of the loop when the user enters a valid input and output a valid input
Input:
    [1] input_ori: original input
    [2] type: the type you need get
        type = 1: int
        type = 2: filename
        type = 3: yes or no
        type = 4: float
    [3] max_value: the max number of option, it is necessary when the type is 1 (default 9)
    [4] path: the path of file, it is necessary when th type is 2 (default "")
Output:
    a valid input
'''

def change_print_color(s):
    return "\033[31m" + s + "\033[0m"

# 判断循环(包括合法性、选项存在与否、文件存在与否）
def loop_legal(input_ori, type, max_value = 9, path = ""):
    # 判断输入格式是否有误
    while (input_legal(input_ori, type) == False):
        if type == 1:
            print(change_print_color("[ERROR] Your input format is incorrect, please re-enter a number (example : 1) :"))
        elif type == 2:
            print(change_print_color("[ERROR] Your input format is incorrect, please re-enter a filename (example : sample.txt) :"))
        elif type == 3:
            print(change_print_color("[ERROR] Your input format is incorrect, please re-enter 'y' or 'n' (example : y) :"))
        elif type == 4:
            print(change_print_color("[ERROR] Your input format is incorrect, please re-enter a float (example : 0.001) :"))
        input_now = input()
        if input_legal(input_now, type) == True:
            input_ori = input_now
            break
        else:
            continue
    # 判断是否有该选项
    if type == 1:
        while (int(input_ori) > int(max_value) or int(input_ori) <=int(0)):
            print(change_print_color("[ERROR] this option does not exist or your input format is incorrect, please re-enter a number (example : 1) :"))
            input_now = input()
            if input_legal(input_now, type) == True:
                if (int(input_now) > int(max_value) or int(input_now) <=int(0)):
                    continue
                else:
                    input_ori = input_now
                    break
        return int(input_ori)
    # 判断是否存在该文件
    elif type == 2:
        while os.path.exists(path + input_ori) == False:
            print(change_print_color("[ERROR] this file does not exist or your input format is incorrect, please re-enter a filename  (example : sample.txt) :"))
            input_now = input()
            if input_legal(input_now, type) == True:
                if os.path.exists(path + input_now) == False:
                    continue
                else:
                    input_ori = input_now
                    break
        return input_ori
    # 判断输入是否为yes或no
    elif type == 3:
        while input_ori != "y" and input_ori != "n":
            print(change_print_color("[ERROR] this option does not exist or your input format is incorrect, please re-enter 'y' or 'n'  (example : y) :"))
            input_now = input()
            if input_legal(input_now, type) == True:
                if input_now != "y" and input_now != "n":
                    continue
                else:
                    input_ori = input_now
                    break
        return input_ori
    # 判断浮点数float是否为0-1
    elif type == 4:
        while (float(input_ori) > float(max_value) or float(input_ori) <= float(0.0)):
            print(change_print_color("[ERROR] this option does not exist or your input format is incorrect, please re-enter a float  (example : 0.001) :"))
            input_now = input()
            if input_legal(input_now, type) == True:
                if (float(input_now) > float(max_value) or float(input_now) <=float(0.0)):
                    continue
                else:
                    input_ori = input_now
                    break
        return float(input_ori)

# 判断输入合法性
def input_legal(input_ori, type):
    legal = False
    # 当需要输入int类型时
    if type == 1:
        legal = input_ori.isdigit()
        return legal

    # 当需要输入文件名
    elif type == 2:
        matchObj = re.match(r'\w+.txt', input_ori)
        if matchObj != None:
            legal = True
            return legal
        else:
            return legal

    # 当需要输入yes或no
    elif type == 3:
        matchObj = re.match(r'[a-z]', input_ori)
        if matchObj != None:
            legal = True
            return legal
        else:
            return legal

    # 当需要输入浮点数
    elif type == 4:
        matchObj = re.match(r'[0-9]+.+[0-9]+', input_ori)
        if matchObj != None:
            legal = True
            return legal
        else:
            return legal
