import torch
import torch.nn as nn
import torch.optim as optim
import numpy
import sys
sys.path.append("..") 

class Interface(object):
    def __init__(self):
        pass
    
    def process(self):
        pass

    def data_process(self):
        pass
    
    def model(self):
        pass
    
    def optimization(self):
        pass

    def evaluation(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass
    
    def test(self):
        pass