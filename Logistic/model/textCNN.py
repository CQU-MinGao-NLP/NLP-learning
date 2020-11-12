import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, embedding_size, sequence_length, num_classes, filter_sizes, num_filters, vocab_size):
        self.embedding_size = embedding_size
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters

        super(TextCNN, self).__init__()
        self.num_filters_total = num_filters * len(filter_sizes)
        self.W = nn.Embedding(vocab_size, embedding_size)
        self.Weight = nn.Linear(self.num_filters_total, num_classes, bias=False)
        self.Bias = nn.Parameter(torch.ones([num_classes]))
        # nn.Conv2d([ channels, output, height_2, width_2 ])
        self.filter_list = nn.ModuleList([nn.Conv2d(1, num_filters, (size, embedding_size)) for size in filter_sizes])

    def forward(self, X):
        embedded_chars = self.W(X) # [batch_size, sequence_length, sequence_length]
        embedded_chars = embedded_chars.unsqueeze(1) # add channel(=1) [batch, channel(=1), sequence_length, embedding_size]

        pooled_outputs = []
        for i, conv in enumerate(self.filter_list):
            # conv : [input_channel(=1), output_channel(=3), (filter_height, filter_width), bias_option]
            h = F.relu(conv(embedded_chars))
            # mp : ((filter_height, filter_width))
            mp = nn.MaxPool2d((self.sequence_length - self.filter_sizes[i] + 1, 1))
            # pooled : [batch_size(=6), output_height(=1), output_width(=1), output_channel(=3)]
            #print("h",h.size())
            #print("mp", mp.size())
            pooled = mp(h).permute(0, 3, 2, 1)
            #print("pooled", pooled.size())
            pooled_outputs.append(pooled)

        h_pool = torch.cat(pooled_outputs, len(self.filter_sizes)) # [batch_size(=6), output_height(=1), output_width(=1), output_channel(=3) * 3]
        #print("h_pool", h_pool.size())
        h_pool_flat = torch.reshape(h_pool, [-1, self.num_filters_total]) # [batch_size(=6), output_height * output_width * (output_channel * 3)]
        #print("h_pool_flat", h_pool_flat.size())
        model = self.Weight(h_pool_flat) + self.Bias # [batch_size, num_classes]
        return model