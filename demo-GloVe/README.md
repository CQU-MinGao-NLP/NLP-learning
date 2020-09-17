demo来自Stanford CS224N第一课。
主要展示了GloVe模型学习到的词向量具有强大的语义。
GloVe == Global Vectors，因训练过程中使用了基于整个数据集计算得到的全局统计信息，故得此名。
demo使用了预训练好的[GloVe](https://github.com/stanfordnlp/GloVe)模型，演示了以下几个应用：
1.找到目标词的最相似的词。
2.找到目标词不相似的词。
3.基于1和2，衍生出一种“类比”的高级玩法。如***"king" - "man" + "woman" = "queen"***、***"fantastic" - "good" + "bad" = "terrible"***。
4.挑选出不属于一组词中不属于该类的词。
5.使用PCA将词向量降到2D，可视化地展示了词向量的聚类。