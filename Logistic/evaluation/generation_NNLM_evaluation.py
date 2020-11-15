import torch
from Logistic.dataprocess import Data_process
import sys
from tqdm import tqdm
import os
import fasttext

from Logistic.dataprocess.Data_process import en_tokenizer

sys.path.append("..")
MODEL_ROOT = "../Data/model/NNLM_predict_next_word/"
RESULT_ROOT = "../Data/result/NNLM_predict_next_word/"

def test(text):
    n_step = 5
    n_words = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sentences = text
    text = en_tokenizer(line.strip() for line in text)
    print(text)
    NNLM_model = torch.load(MODEL_ROOT + 'NNLM_predict_next_word.pkl')

    id2en = Data_process.dict_load(MODEL_ROOT + "NNLM_predict_next_word_idx2token.txt")
    en2id = Data_process.dict_load(MODEL_ROOT + "NNLM_predict_next_word_token2idx.txt")

    input_data = []
    #en_num_data = [[int(en2id[en]) for en in line if en in en2id] for line in en_token_list]
    for line in text:
        length = len(line)
        if length < (n_step):
            continue
        input_data.append(line[-5:])

    result = []
    #for i in range(len(en_num_data)):  
        # en_tokens = list(filter(lambda x: x != 0, en_num_data[i]))  # 过滤零
        # ch_tokens = list(filter(lambda x: x != 3 and x != 0, self.ch_num_data[i]))  # 原文、和机器翻译作对照
        # sentence = [self.id2en[t] for t in en_tokens]
        #print("【原文】")
        # print("".join(sentence))
        # translation = [self.id2ch[t] for t in ch_tokens]
        # print("【原文】")
        # print("".join(translation))
        # test_sample = {}
        # test_sample["src"] = torch.tensor(en_tokens, dtype=torch.long, device=device).reshape(-1, 1)
        # test_sample["src_len"] = [len(en_tokens)]
        
        # seq2seq_model.predict = True
        # seq2seq_model.eval()
        # output_tokens = seq2seq_model(test_sample["src"], test_sample["src_len"])
        # output_tokens = [id2ch[str(t)] for t in output_tokens]
    for id, data in enumerate(input_data):
        text1 = []
        for word in data:
            if word in en2id:
                text1.append(int(en2id[word]))
            else:
                text1.append(0)

        for i in range(n_words):
            predict = NNLM_model(torch.tensor(torch.tensor([text1[-5:]]))).data.max(1, keepdim=True)[1]
            print(predict[0][0].item())
            text1.append(predict[0][0].item())
        result.append(sentences[id].strip() + ' -> ' + " ".join([id2en[str(n)] for n in text1[n_step+1:]]))
        
        #result.append("".join(output_tokens))
        # translate(self.model, test_sample, self.id2ch), end="\n\n"
        
    if os.path.exists(RESULT_ROOT) == False:
        os.mkdir(RESULT_ROOT)
    with open(RESULT_ROOT + "generation_result.txt", 'w', encoding='UTF-8') as f:
        pbar = tqdm(total=len(result))
        for id, line in enumerate(result):
            pbar.update(1)           
            # f.write(sentences[id].strip())
            # f.write(' ')
            f.write(line)
            
            f.write('\n')

        pbar.close()

    print("The embedding of words will be save in " + RESULT_ROOT + "generation_result.txt")

