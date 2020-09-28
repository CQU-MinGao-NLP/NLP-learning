import os
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

path=os.getcwd()
glove_file = datapath(os.path.join(path, "glove.6B.100d.txt"))
word2vec_glove_file = get_tmpfile("glove.6B.100d.word2vec.txt")
glove2word2vec(glove_file, word2vec_glove_file)

model = KeyedVectors.load_word2vec_format(word2vec_glove_file)

# 1
print(model.most_similar('obama'))
# 2
print(model.most_similar(negative='banana'))

# 3
# result = model.most_similar(positive=['woman', 'king'], negative=['man'])
# print("{}: {:.4f}".format(*result[0]))

def analogy(x1, x2, y1):
    result = model.most_similar(positive=[y1, x2], negative=[x1])
    return result[0][0]

print(analogy('japan', 'japanese', 'australia'))
print(analogy('man', 'king', 'woman'))
print(analogy('australia', 'beer', 'france'))
print(analogy('obama', 'clinton', 'reagan'))
print(analogy('tall', 'tallest', 'long'))
print(analogy('good', 'fantastic', 'bad'))

# 4
print(model.doesnt_match("breakfast cereal dinner lunch".split()))

# 5
def display_pca_scatterplot(model, words=None, samlpe=0):
    if words == None:
        if samlpe > 0:
            words = np.random.choice(list(model.vocab.keys()), samlpe)
        else:
            words = [word for word in model.vocab]
    
    word_vectors = np.array([model[w] for w in words])
    
    twodim = PCA().fit_transform(word_vectors)[:, :2]
    plt.figure(figsize=(6, 6))
    plt.scatter(twodim[:, 0], twodim[:, 1], edgecolors='k', c='r')
    for word, (x, y) in zip(words, twodim):
        plt.text(x+0.05, y+0.05, word)
    plt.show()

display_pca_scatterplot(model, ['coffee', 'tea', 'beer', 'wine', 'brandy', 'rum', 'champagne', 'spaghetti', 'borscht', 'hamburger',
                                'pizza', 'falafel', 'meatballs', 'sushi', 'dog', 'horse', 'cat', 'monkey', 'parrot', 'koala', 'frog', 'toad',
                                'monkey', 'ape', 'kangaroo', 'wombat', 'wolf', 'france', 'germany', 'china', 'hungary', 'luxembourg',
                                'australia', 'homework', 'assignment', 'problem', 'exam', 'class', 'test', 'school', 'college',
                                'university', 'institute'])