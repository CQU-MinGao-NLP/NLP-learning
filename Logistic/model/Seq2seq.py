import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy
import torch.nn.functional as F


#seq2seq模型
class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 device,
                 predict=False,
                 basic_dict=None,
                 max_len=100
                 ):
        super(Seq2Seq, self).__init__()

        self.device = device

        self.encoder = encoder
        self.decoder = decoder

        self.predict = predict  # 训练阶段还是预测阶段
        self.basic_dict = basic_dict  # decoder的字典，存放特殊token对应的id
        self.max_len = max_len  # 翻译时最大输出长度

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        assert encoder.gru.bidirectional == decoder.gru.bidirectional, \
            "Decoder and encoder must had same value of bidirectional attribute!"

    def forward(self, input_batches, input_lengths, target_batches=None, target_lengths=None,
                teacher_forcing_ratio=0.5):
        # input_batches = [seq_len, batch]
        # target_batches = [seq_len, batch]
        batch_size = input_batches.size(1)

        BOS_token = self.basic_dict["<bos>"]
        EOS_token = self.basic_dict["<eos>"]
        PAD_token = self.basic_dict["<pad>"]

        # 初始化
        enc_n_layers = self.encoder.gru.num_layers
        enc_n_directions = 2 if self.encoder.gru.bidirectional else 1
        encoder_hidden = torch.zeros(enc_n_layers * enc_n_directions, batch_size, self.encoder.hid_dim,
                                     device=self.device)

        # encoder_outputs = [input_lengths, batch, hid_dim * n directions]
        # encoder_hidden = [n_layers*n_directions, batch, hid_dim]
        encoder_outputs, encoder_hidden = self.encoder(
            input_batches, input_lengths, encoder_hidden)

        # 初始化
        decoder_input = torch.tensor([BOS_token] * batch_size, dtype=torch.long, device=self.device)
        decoder_hidden = encoder_hidden

        if self.predict:
            # 一次只输入一句话
            assert batch_size == 1, "batch_size of predict phase must be 1!"
            output_tokens = []

            while True:
                decoder_output, decoder_hidden, decoder_attn = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # [1, 1]
                topv, topi = decoder_output.topk(1)   # 返回张量中最大值
                decoder_input = topi.squeeze(1).detach()
                output_token = topi.squeeze().detach().item()
                if output_token == EOS_token or len(output_tokens) == self.max_len:
                    break
                output_tokens.append(output_token)
            return output_tokens

        else:
            max_target_length = max(target_lengths)
            all_decoder_outputs = torch.zeros((max_target_length, batch_size, self.decoder.output_dim),
                                              device=self.device)

            for t in range(max_target_length):
                # Teacher Forcing的思想就是两者之间，有一定概率用上一次预测的词或者是上一个正确的词作为输入。
                use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
                if use_teacher_forcing:
                    # decoder_output = [batch, output_dim]
                    # decoder_hidden = [n_layers*n_directions, batch, hid_dim]
                    decoder_output, decoder_hidden, decoder_attn = self.decoder(
                        decoder_input, decoder_hidden, encoder_outputs
                    )
                    all_decoder_outputs[t] = decoder_output
                    decoder_input = target_batches[t]  # 下一个输入来自训练数据
                else:
                    decoder_output, decoder_hidden, decoder_attn = self.decoder(
                        decoder_input, decoder_hidden, encoder_outputs
                    )
                    # [batch, 1]
                    topv, topi = decoder_output.topk(1)
                    all_decoder_outputs[t] = decoder_output
                    decoder_input = topi.squeeze(1).detach()  # 下一个输入来自模型预测

            # NLLLoss适用于最后一层为logsoftmax的网络
            loss_fn = nn.NLLLoss(ignore_index=PAD_token)
            loss = loss_fn(
                all_decoder_outputs.reshape(-1, self.decoder.output_dim),  # [batch*seq_len, output_dim]
                target_batches.reshape(-1)  # [batch*seq_len]，不分行和列
            )
            return loss

# Encoder函数
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout=0.5, bidirectional=True):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(self.input_dim, self.emb_dim)
        # 每个单元为GRU代替原始RNN，
        # emb_dim: 单个GRU输入嵌入维度     hid_dim：单个GRU输出维度   n_layers: GRU的层数    如果双向，输出增加一倍
        self.gru = nn.GRU(self.emb_dim, self.hid_dim, self.n_layers, dropout=self.dropout, bidirectional=self.bidirectional)

    def forward(self, input_seqs, input_lengths, hidden):
        # input_seqs = [seq_len, batch]
        embedded = self.embedding(input_seqs)
        # embedded = [seq_len, batch, embed_dim]
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, enforce_sorted=False)

        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # outputs = [seq_len, batch, hid_dim * n directions]
        # output_lengths = [batch]
        return outputs, hidden

# 选择attention计算方法：dot、general、concat
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)  # [seq_len, batch]   对某维度求和

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)  # [seq_len, batch, hid_dim]
        return torch.sum(hidden * energy, dim=2)  # [seq_len, batch]

    def concat_score(self, hidden, encoder_output):
        # hidden.expand(encoder_output.size(0), -1, -1) -> [seq_len, batch, N]
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        # energy = [sql_len, batch, hidden_size]
        return torch.sum(self.v * energy, dim=2)  # [seq_len, batch]

    def forward(self, hidden, encoder_outputs):
        # hidden = [1, batch,  n_directions * hid_dim]
        # encoder_outputs = [seq_len, batch, hid dim * n directions]
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        attn_energies = attn_energies.t()  # [batch, seq_len]   .t()转置函数

        return F.softmax(attn_energies, dim=1).unsqueeze(1)  # softmax归一化# [batch, 1, seq_len] unsqueeze（arg）是增添第arg个维度为1

# 带有注意力机制的Decoder函数
class AttnDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers=1, dropout=0.5, bidirectional=True,
                 attn_method="general"):
        super(AttnDecoder, self).__init__()

        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, bidirectional=bidirectional)

        if bidirectional:
            self.concat = nn.Linear(hid_dim * 2 * 2, hid_dim * 2)
            self.out = nn.Linear(hid_dim * 2, output_dim)
            self.attn = Attn(attn_method, hid_dim * 2)
        else:
            self.concat = nn.Linear(hid_dim * 2, hid_dim)
            self.out = nn.Linear(hid_dim, output_dim)
            self.attn = Attn(attn_method, hid_dim)
        self.softmax = nn.LogSoftmax(dim=1)  # 比softmax前多加log，因此输出为负

    def forward(self, token_inputs, last_hidden, encoder_outputs):
        batch_size = token_inputs.size(0)
        embedded = self.embedding(token_inputs)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(1, batch_size, -1)  # [1, B, hid_dim]

        gru_output, hidden = self.gru(embedded, last_hidden)
        # gru_output = [1, batch,  n_directions * hid_dim]
        # hidden = [n_layers * n_directions, batch, hid_dim]

        # encoder_outputs = [sql_len, batch, hid dim * n directions]
        attn_weights = self.attn(gru_output, encoder_outputs)
        # attn_weights = [batch, 1, sql_len]
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))    # bmm，tensor矩阵乘法
        # [batch, 1, hid_dim * n directions]

        # LuongAttention
        # squeeze（0）函数表示第一个维度为1则去除第一维度
        gru_output = gru_output.squeeze(0)  # [batch, n_directions * hid_dim]
        context = context.squeeze(1)  # [batch, n_directions * hid_dim]
        # 拼接张量，按第二维
        concat_input = torch.cat((gru_output, context), 1)  # [batch, n_directions * hid_dim * 2]
        concat_output = torch.tanh(self.concat(concat_input))  # [batch, n_directions*hid_dim]

        output = self.out(concat_output)  # [batch, output_dim]
        output = self.softmax(output)

        return output, hidden, attn_weights