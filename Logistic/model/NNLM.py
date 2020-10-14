import torch
import torch.nn as nn
import torch.optim as optim
import numpy

class NNLM(nn.Module):
    def __init__(self, n_class, n_step, n_hidden, m):
        self.n_class = n_class
        self.n_step = n_step
        self.n_hidden = n_hidden
        self.m = m
        
        super(NNLM, self).__init__()
        self.C = nn.Embedding(n_class, m)
        self.H = nn.Linear(n_step * m, n_hidden, bias=False)
        self.d = nn.Parameter(torch.ones(n_hidden))
        self.U = nn.Linear(n_hidden, n_class, bias=False)
        self.W = nn.Linear(n_step * m, n_class, bias=False)
        self.b = nn.Parameter(torch.ones(n_class))

    def forward(self, X):
        X = self.C(X) # X : [batch_size, n_step, n_class]
        X = X.view(-1, self.n_step * self.m) # [batch_size, n_step * n_class]
        tanh = torch.tanh(self.d + self.H(X)) # [batch_size, n_hidden]
        output = self.b + self.W(X) + self.U(tanh) # [batch_size, n_class]
        return output