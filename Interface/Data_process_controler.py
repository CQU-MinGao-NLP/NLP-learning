import os
import sys

sys.path.append("..")
from Logistic.dataprocess import Data_process

class Data_process_controler(object):
    def __init__(self, filename, method):
        self.input = []
        self.filename = filename
        self.method_name = method
        self.output = []
        
    def process(self):
        self.load()
        self.method()
        self.save()

    def load(self):
        self.input = Data_process.read_file("../Data/raw_data/"+self.filename)
        print("read data succeed！")
        
    def method(self):
        """
        print("1. en_tokenizer  2. cn_tokenizer     3. en_stopwords")
        print("4. cn_stopwords  5. filter_lowfrequency  6. filter_html ")
        print("7. stemming")
        """
        if self.method_name == "en_tokenizer":
            self.output.append(Data_process.en_tokenizer(self.input))
        elif self.method_name == "cn_tokenizer":
            self.output.append(Data_process.cn_tokenizer(self.input))
        elif self.method_name == "en_stopwords":
            self.output.append(Data_process.en_stopwords(self.input))
        elif self.method_name == "cn_stopwords":
            self.output.append(Data_process.cn_stopwords(self.input))
        elif self.method_name == "filter_lowfrequency":
            print("please give the frequency limitation:(like 5)")
            num = int(input())
            self.output.append(Data_process.filter_lowfrequency(self.input, num))
        elif self.method_name == "filter_html":
            self.output.append(Data_process.filter_html(self.input))
        elif self.method_name == "stemming":
            self.output.append(Data_process.stemming(self.input))

    def save(self):
        path = "../Data/output/processed_"+self.method_name+ "_"+self.filename
        num = 1
        while os.path.exists(path) == True:
            path = path[:-4] + str(num) + path[-4:]
            num += 1
        if len(self.output) == 1:
            Data_process.save_file(self.output, path)
        elif len(self.output) == 2:
            Data_process.save_file(self.output[1], path)
        else:
            print("Error input!")
            exit(-1)
        print("The file has been saved to " + path[3:])