import collections
import math
import random
import sys
import time
import os
import numpy as np
import torch
from torch import nn
import torch.utils.data as Data
from Interface import Interface
from Logistic.dataprocess import Data_process
from Logistic.dataprocess.Data_process import dict_save

sys.path.append("..")

'''
功能：  
    使用word2vec模型产生词向量
输入（可调整的参数）：  
    [1]embedding_size = 100 # embedding size  词向量大小
    [2]max_window_size = 5 # 最大窗口长度
    [3]just_test = False # 是否只用于预测
'''
MODEL_ROOT = "../Data/model/word2vec/"
DATA_ROOT = "../Data/train_data/"

class word2vec(Interface.Interface):
    def __init__(self, filename, embedding_size=100, learning_rate = 0.01, num_epochs = 10, batch_size = 512, max_window_size = 5):
        #super(Interface, self).__init__()
        self.filename = filename
        self.input_data = []
        self.input_labels = []
        self.embedding_size = embedding_size
        self.learning_rate = 0.01
        self.num_epochs = 10
        self.batch_size = 512
        self.max_window_size = 5

    # 控制流程
    def process(self):
        self.data_process()
        self.make_batch()
        self.model()
        self.optimization()
        self.train()
        #self.predict()
        #self.test()

    def make_batch(self):
        class MyDataset(torch.utils.data.Dataset):
            def __init__(self, centers, contexts, negatives):
                assert len(centers) == len(contexts) == len(negatives)
                self.centers = centers
                self.contexts = contexts
                self.negatives = negatives

            def __getitem__(self, index):
                return (self.centers[index], self.contexts[index], self.negatives[index])

            def __len__(self):
                return len(self.centers)

        def batchify(data):
            max_len = max(len(c) + len(n) for _, c, n in data)
            centers, contexts_negatives, masks, labels = [], [], [], []
            for center, context, negative in data:
                cur_len = len(context) + len(negative)
                centers += [center]
                contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
                masks += [[1] * cur_len + [0] * (max_len - cur_len)]
                labels += [[1] * len(context) + [0] * (max_len - len(context))]
            return (torch.tensor(centers).view(-1, 1), torch.tensor(contexts_negatives),
                    torch.tensor(masks), torch.tensor(labels))

        
        num_workers = 0 if sys.platform.startswith('win32') else 4

        dataset = MyDataset(self.all_centers, 
                            self.all_contexts, 
                            self.all_negatives)
        self.data_iter = Data.DataLoader(dataset, self.batch_size, shuffle=True,
                                    collate_fn=batchify, 
                                    num_workers=num_workers)

    def data_process(self):
        # attention
        assert self.filename in os.listdir(DATA_ROOT)

        with open(DATA_ROOT+self.filename, 'r') as f:
            lines = f.readlines()
            raw_dataset = [st.split() for st in lines]

        counter = Data_process.highfrequency(raw_dataset, 5)[0]

        # counter = collections.Counter([tk for st in raw_dataset for tk in st])
        # counter = dict(filter(lambda x: x[1] >= 5, counter.items()))

        # ！！！！！！！！！！！！！！！
        self.idx_to_token = [tk for tk, _ in counter.items()]
        self.token_to_idx = {tk: idx for idx, tk in enumerate(self.idx_to_token)}
        dict_save(dict(zip(list(range(len(self.idx_to_token))), self.idx_to_token)), MODEL_ROOT + "word2vec_idx2token.txt")
        dict_save(self.token_to_idx, MODEL_ROOT + "word2vec_token2idx.txt")


        dataset = [[self.token_to_idx[tk] for tk in st if tk in self.token_to_idx]
                for st in raw_dataset]
        num_tokens = sum([len(st) for st in dataset])

        def discard(idx):
            return random.uniform(0, 1) < 1 - math.sqrt(
                1e-4 / counter[self.idx_to_token[idx]] * num_tokens)

        subsampled_dataset = [[tk for tk in st if not discard(tk)] for st in dataset]


        def get_centers_and_contexts(dataset, max_window_size):
            centers, contexts = [], []
            for st in dataset:
                if len(st) < 2:  
                    continue
                centers += st
                for center_i in range(len(st)):
                    window_size = random.randint(1, max_window_size)
                    indices = list(range(max(0, center_i - window_size),
                                        min(len(st), center_i + 1 + window_size)))
                    indices.remove(center_i)  
                    contexts.append([st[idx] for idx in indices])
            return centers, contexts

        self.all_centers, self.all_contexts = get_centers_and_contexts(subsampled_dataset, self.max_window_size)

        def get_negatives(all_contexts, sampling_weights, K):
            all_negatives, neg_candidates, i = [], [], 0
            population = list(range(len(sampling_weights)))
            for contexts in all_contexts:
                negatives = []
                while len(negatives) < len(contexts) * K:
                    if i == len(neg_candidates):
                        i, neg_candidates = 0, random.choices(
                            population, sampling_weights, k=int(1e5))
                    neg, i = neg_candidates[i], i + 1
                    
                    if neg not in set(contexts):
                        negatives.append(neg)
                all_negatives.append(negatives)
            return all_negatives

        sampling_weights = [counter[w]**0.75 for w in self.idx_to_token]
        self.all_negatives = get_negatives(self.all_contexts, sampling_weights, 5)


    def model(self):
        self.word2vec_net = nn.Sequential(
            nn.Embedding(num_embeddings=len(self.idx_to_token), embedding_dim=self.embedding_size),
            nn.Embedding(num_embeddings=len(self.idx_to_token), embedding_dim=self.embedding_size)
        )

    def optimization(self):
        class SigmoidBinaryCrossEntropyLoss(nn.Module):
            def __init__(self):
                super(SigmoidBinaryCrossEntropyLoss, self).__init__()
            def forward(self, inputs, targets, mask=None):
                inputs, targets, mask = inputs.float(), targets.float(), mask.float()
                res = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none", weight=mask)
                return res.mean(dim=1)

        self.loss = SigmoidBinaryCrossEntropyLoss()

    def train(self):
        def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
            v = embed_v(center)
            u = embed_u(contexts_and_negatives)
            pred = torch.bmm(v, u.permute(0, 2, 1))
            return pred
        print('start train!')
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("train on", device)
        self.word2vec_net = self.word2vec_net.to(device)
        optimizer = torch.optim.Adam(self.word2vec_net.parameters(), lr=self.learning_rate)
        for epoch in range(self.num_epochs):
            start, l_sum, n = time.time(), 0.0, 0
            for batch in self.data_iter:
                center, context_negative, mask, label = [d.to(device) for d in batch]

                pred = skip_gram(center, context_negative, self.word2vec_net[0], self.word2vec_net[1])

                l = (self.loss(pred.view(label.shape), label, mask) *
                    mask.shape[1] / mask.float().sum(dim=1)).mean()
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                l_sum += l.cpu().item()
                n += 1
            print('epoch %d, loss %.2f, time %.2fs'
                % (epoch + 1, l_sum / n, time.time() - start))

        torch.save(self.word2vec_net, MODEL_ROOT+'word2vec.pkl')

    def predict(self):
        pass
    
    def test(self):
        self.word2vec_net.load_state_dict(torch.load(MODEL_ROOT+'word2vec.pkl'))
        def get_similar_tokens(query_token, k, embed):
            W = embed.weight.data
            x = W[self.token_to_idx[query_token]]
            cos = torch.matmul(W, x) / (torch.sum(W * W, dim=1) * torch.sum(x * x) + 1e-9).sqrt()
            _, topk = torch.topk(cos, k=k+1)
            topk = topk.cpu().numpy()
            for i in topk[1:]:
                print('cosine sim=%.3f: %s' % (cos[i], (self.idx_to_token[i])))

        get_similar_tokens('chip', 3, self.word2vec_net[0])