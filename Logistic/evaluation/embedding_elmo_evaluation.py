from Logistic.dataprocess.Data_process import *
from elmoformanylangs import Embedder



MODEL_ROOT = "../Data/model/elmo/"
RESULT_ROOT = "../Data/result/elmo/"

# 0 for the word encoder
# 1 for the first LSTM hidden layer
# 2 for the second LSTM hidden layer
# -1 for an average of 3 layers. (default)
# -2 for all 3 layers

def test(text):
    sentences = text
    e = Embedder(MODEL_ROOT + 'zhs.model/')
    text = cn_tokenizer(text)
    result = e.sents2elmo(text, output_layer=-1)

    if os.path.exists(RESULT_ROOT) == False:
        os.makedirs(RESULT_ROOT)
        # os.mkdir(RESULT_ROOT)
    with open(RESULT_ROOT + "word_embedding.txt", 'w', encoding='UTF-8') as f:
        pbar = tqdm(total=len(text))
        for i in range(len(text)):
            pbar.update(1)
            f.write(sentences[i])
            for id, word in enumerate(text[i]):
                f.write(text[i][id])
                f.write(' ')
                for elem in result[i][id]:
                    f.write(str(elem))
                    f.write(' ')
                f.write('\n')
            f.write('#'*80)
        pbar.close()
    print("The embedding of words will be save in " + RESULT_ROOT + "word_embedding.txt")

