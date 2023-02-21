import os
from gensim.models import Word2Vec
import gensim
from gensim.models import KeyedVectors
import jieba
import pickle as pkl
import sys


"""
使用 Word2Vec 训练字向量、词向量并使用
"""


def train_single_word(f_text, f_out_wordVec, f_out_wordModel, f_wordPartialWeight):
    # jy: 获取文本序列列表;
    datas = open(f_text, 'r').read().split("\n")
    # jy: 对每一行文本序列进行字符级别的 token 获取
    word_datas = [[i for i in data if i != " "] for data in datas]

    model = Word2Vec(
        word_datas,       # 需要训练的文本
        vector_size=10,   # 词向量的维度
        window=2,         # 句子中当前单词和预测单词之间的最大距离
        min_count=1,      # 忽略总频率低于此的所有单词 出现的频率小于 min_count 不用作词向量
        workers=8,        # 使用这些工作线程来训练模型（使用多核机器进行更快的训练）
        sg=0,             # 训练方法: 1 表示 skip-gram, 0 表示 CBOW
        epochs=10         # 语料库上的迭代次数
    )

    # 字向量保存; binary 如果为 True, 则数据将以二进制 word2vec 格式保存, 否则将以纯文本格式保存
    model.wv.save_word2vec_format(f_out_wordVec, binary=False)

    # 模型保存
    model.save(f_out_wordModel)

    # 模型中的 wv 和 syn1neg 都可以单独保存
    pkl.dump([model.wv.index_to_key, model.wv.key_to_index, model.wv.vectors],
             open(f_wordPartialWeight, "wb"))


def use_single_wordVec_model(f_word_model, f_word_vector, single_word="提"):
    # jy: 方式 1: 通过模型加载词向量 (推荐)
    model = gensim.models.Word2Vec.load(f_word_model)
    ls_ = model.wv.index_to_key
    #print(ls_)
    print(len(ls_))
    print("通过模型查看字向量：", model.wv[single_word])

    # jy: 方式 2: 通过字向量加载
    vector = KeyedVectors.load_word2vec_format(f_word_vector)
    # jy: 返回前 3 个相似键的数量
    ls_top = vector.most_similar(single_word, topn=3)
    print("与 %s 含义相似的字是: " % single_word, ls_top)
    print("通过字向量查看字向量：", vector[single_word])



def train_words(f_text, f_out_wordsVec, f_out_wordsModel, f_wordsPartialWeight):
    datas = open(f_text, "r").read().split("\n")
    # jy: 需使用分词工具, 此处中文, 用 jieba 分词;
    words_datas = [[i for i in (jieba.cut(data)) if i != " "] for data in datas]
    model = Word2Vec(
        words_datas,      # 需要训练的文本
        vector_size=10,   # 词向量的维度
        window=2,         # 句子中当前单词和预测单词之间的最大距离
        min_count=1,      # 忽略总频率低于此的所有单词 出现的频率小于 min_count 不用作词向量
        workers=8,        # 使用这些工作线程来训练模型（使用多核机器进行更快的训练）
        sg=0,             # 训练方法: 1 表示 skip-gram, 0 表示 CBOW
        epochs=10         # 语料库上的迭代次数
    )
    model.wv.save_word2vec_format(f_out_wordsVec, binary=False)

    model.save(f_out_wordsModel)
    pkl.dump([model.wv.index_to_key, model.wv.key_to_index, model.wv.vectors],
             open(f_wordsPartialWeight, "wb"))


def use_wordsVec(f_words_model, f_words_vector, words="提到"):
    # 方式1: 通过模型加载词向量 (推荐)
    model = gensim.models.Word2Vec.load(f_words_model)

    dic = model.wv.key_to_index
    print(dic)
    print(len(dic))

    print("通过模型进行查看：", model.wv[words])

    # 2 通过词向量加载
    vector = KeyedVectors.load_word2vec_format(f_words_vector)
    ls_top = vector.most_similar(words, topn=3)
    print("与 %s 含义相似的词是: " % words, ls_top)
    print("通过字向量进行查看：", vector[words])



def read_partial_weight(f_partialWeight):
    dataset = pkl.load(open(f_partialWeight, "rb"))
    index_to_key, key_to_index,  vector = dataset[0], dataset[1], dataset[2]
    print("index_to_key:", index_to_key)
    print("key_to_index:", key_to_index)
    print("vector:", vector)



if __name__ == "__main__":
    f_text = "train.txt"

    print("=" * 66)
    print("字向量训练和使用")
    print("=" * 66)
    f_out_wordVec = "out_singleWord/word_data.vector"
    f_out_wordModel = "out_singleWord/word.model"
    f_wordPartialWeight = "out_singleWord/WordPartialWeight.pkl"
    # jy: 训练字向量;
    #train_single_word(f_text, f_out_wordVec, f_out_wordModel, f_wordPartialWeight)
    # jy: 使用字向量;
    #use_single_wordVec_model(f_out_wordModel, f_out_wordVec)
    # jy: 查看 partial weight 文件内容;
    read_partial_weight(f_wordPartialWeight)


    print("=" * 66)
    print("字向量训练和使用")
    print("=" * 66)
    f_out_wordsVec = "out_words/words_data.vector"
    f_out_wordsModel = "out_words/words.model"
    f_wordsPartialWeight = "out_words/WordsPartialWeight.pkl"
    # jy: 训练词向量;
    #train_words(f_text, f_out_wordsVec, f_out_wordsModel, f_wordsPartialWeight)
    # jy: 使用词向量;
    #use_wordsVec(f_out_wordsModel, f_out_wordsVec)
    # jy: 查看 partial weight 文件内容;
    #read_partial_weight(f_wordsPartialWeight)

