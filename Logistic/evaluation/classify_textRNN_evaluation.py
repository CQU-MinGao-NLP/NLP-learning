import torch
from Logistic.dataprocess import Data_process
import sys
from tqdm import tqdm
import os

from Logistic.dataprocess.Data_process import en_tokenizer

sys.path.append("..")
MODEL_ROOT = "../Data/model/textRNN_classify/"
RESULT_ROOT = "../Data/result/textRNN_classify/"
def test(text):
    sentences = text
    text = en_tokenizer(text)
    textRNN_model = torch.load(MODEL_ROOT + 'textRNN_model.pkl')
    token_to_idx = Data_process.dict_load(MODEL_ROOT + "textRNN_token2idx.txt")
    idx_to_token = Data_process.dict_load(MODEL_ROOT + "textRNN_idx2token.txt")

    def predict_sentiment(net, token_to_idx, sentence):
        device = list(net.parameters())[0].device
        sentence = torch.tensor([int(token_to_idx[word]) for word in sentence if word in token_to_idx], device=device)
        label = torch.argmax(net(sentence.view((1, -1))), dim=1)
        return label.item()

    result = []
    for sen in text:
        result.append(str(predict_sentiment(textRNN_model, token_to_idx, sen)))
        #result.append(str(predict_sentiment(textRNN_model, token_to_idx, sen)) + ' ' + str(sen))
        
    if os.path.exists(RESULT_ROOT) == False:
        os.mkdir(RESULT_ROOT)

    with open(RESULT_ROOT + "classify_result.txt", 'w', encoding='UTF-8') as f:
        pbar = tqdm(total=len(result))
        for id, line in enumerate(result):
            pbar.update(1)
            # for i in line:
            #     f.write(i)
            #     f.write(' ')
            f.write(line)
            f.write(' ')
            f.write(sentences[id])


        pbar.close()

    print("The embedding of words will be save in " + RESULT_ROOT + "classify_result.txt")

