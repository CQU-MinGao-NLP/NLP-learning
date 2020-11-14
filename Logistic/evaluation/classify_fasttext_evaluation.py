import torch
from Logistic.dataprocess import Data_process
import sys
from tqdm import tqdm
import os
import fasttext

from Logistic.dataprocess.Data_process import en_tokenizer

sys.path.append("..")
MODEL_ROOT = "../Data/model/fasttext_classify/"
RESULT_ROOT = "../Data/result/fasttext_classify/"

def test(text):

    classifier = fasttext.load_model(MODEL_ROOT + 'fasttext_model_file.bin')
    
    if os.path.exists(RESULT_ROOT) == False:
        os.mkdir(RESULT_ROOT)

    result = []

    for line in text:
        if line == '':
            continue
        
        
        result.append(str(classifier.predict(line.strip())[0][0]) + ' ' + line)
        

    with open(RESULT_ROOT + "classify_result.txt", 'w', encoding='UTF-8') as f:
        pbar = tqdm(total=len(result))
        for line in result:
            pbar.update(1)
            # for i in line:
            #     f.write(i)
            #     f.write(' ')
            f.write(line)
            f.write('\n')

        pbar.close()

    print("The embedding of words will be save in " + RESULT_ROOT + "classify_result.txt")

