import torch
from Logistic.dataprocess import Data_process
import sys
from tqdm import tqdm
import os
import fasttext

from Logistic.dataprocess.Data_process import en_tokenizer

sys.path.append("..")
MODEL_ROOT = "../Data/model/Seq2seq_translate_text/"
RESULT_ROOT = "../Data/result/Seq2seq_translate_text/"

def test(text):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sentences = text
    en_token_list = [[char for char in line] + ["<eos>"] for line in text]

    seq2seq_model = torch.load(MODEL_ROOT + 'Seq2seq_translate_text.pkl')
    en2id = Data_process.dict_load(MODEL_ROOT + "Seq2seq_translate_text_en2id.txt")
    id2en = Data_process.dict_load(MODEL_ROOT + "Seq2seq_translate_text_id2en.txt")
    ch2id = Data_process.dict_load(MODEL_ROOT + "Seq2seq_translate_text_ch2id.txt")
    id2ch = Data_process.dict_load(MODEL_ROOT + "Seq2seq_translate_text_id2ch.txt")

    en_num_data = [[int(en2id[en]) for en in line if en in en2id] for line in en_token_list]
    
    result = []
    for i in range(len(en_num_data)):  
        en_tokens = list(filter(lambda x: x != 0, en_num_data[i]))  # 过滤零
        # ch_tokens = list(filter(lambda x: x != 3 and x != 0, self.ch_num_data[i]))  # 原文、和机器翻译作对照
        # sentence = [self.id2en[t] for t in en_tokens]
        #print("【原文】")
        # print("".join(sentence))
        # translation = [self.id2ch[t] for t in ch_tokens]
        # print("【原文】")
        # print("".join(translation))
        test_sample = {}
        test_sample["src"] = torch.tensor(en_tokens, dtype=torch.long, device=device).reshape(-1, 1)
        test_sample["src_len"] = [len(en_tokens)]
        
        seq2seq_model.predict = True
        seq2seq_model.eval()
        output_tokens = seq2seq_model(test_sample["src"], test_sample["src_len"])
        output_tokens = [id2ch[str(t)] for t in output_tokens]

        result.append("".join(output_tokens))
        # translate(self.model, test_sample, self.id2ch), end="\n\n"
        
    if os.path.exists(RESULT_ROOT) == False:
        os.mkdir(RESULT_ROOT)
    with open(RESULT_ROOT + "translate_result.txt", 'w', encoding='UTF-8') as f:
        pbar = tqdm(total=len(result))
        for id, line in enumerate(result):
            pbar.update(1)           
            f.write(sentences[id].strip())
            f.write(' ')
            f.write(line)
            
            f.write('\n')

        pbar.close()

    print("The embedding of words will be save in " + RESULT_ROOT + "translate_result.txt")

