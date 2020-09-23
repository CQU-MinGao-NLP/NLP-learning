import fasttext.FastText as fasttext


# 模型的训练
def train():
    model = fasttext.train_supervised("./dataset/train.txt", lr=0.1, dim=100, epoch=500, word_ngrams=4, loss='softmax')
    model.save_model("./model/model_file.bin")


# 模型的测试
def test_1():
    classifier = fasttext.load_model("./model/model_file.bin")
    f1 = open('./prediction/prediction.txt', 'w', encoding='utf-8')
    with open('./dataset/test.txt', encoding='utf-8') as fp:
        for line in fp.readlines():
            line = line.strip()
            if line == '':
                continue
            f1.write(line + '\t#####\t' + classifier.predict([line])[0][0][0] + '\n')
    f1.close()


if __name__ == '__main__':
    train()
    test_1()
