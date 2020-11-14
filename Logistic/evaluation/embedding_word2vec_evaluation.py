import torch
from Logistic.dataprocess import Data_process
import sys
from tqdm import tqdm
import os

from Logistic.dataprocess.Data_process import en_tokenizer

sys.path.append("..")
MODEL_ROOT = "../Data/model/word2vec/"
RESULT_ROOT = "../Data/result/word2vec/"

def test(text):
    text = en_tokenizer(text)
    words = Data_process.highfrequency(text, 0)[1]

    word2vec = torch.load(MODEL_ROOT + 'word2vec.pkl')
    token_to_idx = Data_process.dict_load(MODEL_ROOT + "word2vec_token2idx.txt")
    idx_to_token = Data_process.dict_load(MODEL_ROOT + "word2vec_idx2token.txt")
    def get_similar_tokens(query_token, k, embed):
        W = embed.weight.data
        x = W[int(token_to_idx[query_token])]
        cos = torch.matmul(W, x) / (torch.sum(W * W, dim=1) * torch.sum(x * x) + 1e-9).sqrt()
        _, topk = torch.topk(cos, k= k +1)
        topk = topk.cpu().numpy()
        for i in topk[1:]:
            print('cosine sim=%.3f: %s' % (cos[i], (idx_to_token[str(i)])))
    def get_embedding(query_token, embed):
        W = embed.weight.data
        if query_token not in token_to_idx:
            return 0
        x = W[int(token_to_idx[query_token])]
        return x


    if os.path.exists(RESULT_ROOT) == False:
        os.mkdir(RESULT_ROOT)
    with open(RESULT_ROOT + "word_embedding.txt", 'w', encoding='UTF-8') as f:
        pbar = tqdm(total=len(words))
        for word in words:
            pbar.update(1)
            #print(str(cnt+1) + '/' + str(len(words)))
            word_emb = get_embedding(word, word2vec[0])\

            f.write(word)
            f.write(' ')
            f.write(str(word_emb))
            f.write('\n')
        pbar.close()

    print("The embedding of words will be save in " + RESULT_ROOT + "word_embedding.txt")
    # get_similar_tokens('chip', 3, word2vec[0])

