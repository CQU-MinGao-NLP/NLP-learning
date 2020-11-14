import torch
from Logistic.dataprocess import Data_process
import sys
from tqdm import tqdm
import os
import numpy as np

from Logistic.dataprocess.Data_process import en_tokenizer

sys.path.append("..")
MODEL_ROOT = "../Data/model/textCNN_classify/"
RESULT_ROOT = "../Data/result/textCNN_classify/"
def test(text):
    sentences = text
    text = en_tokenizer(text)
    textCNN_model = torch.load(MODEL_ROOT + 'textCNN_model.pkl')
    token_to_idx = Data_process.dict_load(MODEL_ROOT + "textCNN_token2idx.txt")
    idx_to_token = Data_process.dict_load(MODEL_ROOT + "textCNN_idx2token.txt")

    def predict_sentiment(net, token_to_idx, sentence):
        max_l = 500
        def pad(x):
            return x[:max_l] if len(x) > max_l else x + [0] * (max_l - len(x))

        device = list(net.parameters())[0].device
        sentence = torch.tensor([pad([int(token_to_idx[word]) for word in sentence if word in token_to_idx])], device=device)
        label = torch.argmax(net(sentence), dim=1)
        # net(test_batch)
        
        # net = net.to('cpu')
        # test_text = 'i sorry that i hate'
        # tests = [np.asarray(pad([int(token_to_idx[n]) for n in test_text.split()]))]
        # test_batch = torch.LongTensor(tests)
        # predict = net(test_batch)
        return label.item()

    result = []
    for sen in text:
        result.append(str(predict_sentiment(textCNN_model, token_to_idx, sen)))
        
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

