import sys

sys.path.append("..")
from Logistic.dataprocess import Data_process

class Data_process_controler(object):
    def __init__(self, filename, number):
        self.input = []
        self.filename = filename
        self.number = number
        self.output = []
    def process(self):
        self.load()
        self.method()
        self.save()

    def load(self):
        self.input = Data_process.read_file("../Data/raw_data/"+self.filename)
        print("succeed read data！")
        
    def method(self):
        """
        print("1. en_tokenizer  2. cn_tokenizer     3. en_stopwords")
        print("4. cn_stopwords  5. lowfrequency     6. highfrequency")
        print("7. filter_lowfrequency  8. filter_html     9. stemming")
        """
        if self.number == 1:
            self.output.append(Data_process.en_tokenizer(self.input))
        elif self.number == 2:
            self.output.append(Data_process.cn_tokenizer(self.input))
        elif self.number == 3:
            self.output.append(Data_process.en_stopwords(self.input))
        elif self.number == 4:
            self.output.append(Data_process.cn_stopwords(self.input))
        elif self.number == 5:
            print("please give the frequency limitation:(like 5)")
            num = int(input())
            self.output.append(Data_process.lowfrequency(self.input, num))
        elif self.number == 6:
            print("please give the frequency limitation:(like 5)")
            num = int(input())
            self.output.append(Data_process.highfrequency(self.input, num))
        elif self.number == 7:
            print("please give the frequency limitation:(like 5)")
            num = int(input())
            self.output.append(Data_process.filter_lowfrequency(self.input, num))
        elif self.number == 8:
            self.output.append(Data_process.filter_html(self.input))
        elif self.number == 9:
            self.output.append(Data_process.stemming(self.input))

    def save(self):
        print(self.output)
        if len(self.output) == 1:
            Data_process.save_file(self.output, "../Data/output/processed"+self.filename)
        elif len(self.output) == 2:
            Data_process.save_file(self.output[1], "../Data/output/processed"+self.filename)
        else:
            print("Error input!")
            exit(-1)
        print("The file has been saved to Data/output/processed"+self.filename)