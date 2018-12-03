# coding:utf-8
from __future__ import print_function
import sys
import numpy as np
from numpy import zeros
from gensim.models.word2vec import Word2Vec

# 1.扩展 words_index ={word:index}
# 2.扩展 矩阵 按 index 排序
# 3.get vect  by word
# ** 使用时只用到了 初始化词的 索引值
INIT_WORDS = ['<PAD>', '<UNK>', '<GO>', '<EOS>']


class MyWord2vec(object):
    def __init__(self, vec_model_file):
        '''
        :param vec_model_file: 模型文件路径
        :param word_vect_dict: 词向量 dict
        :param word_vect_modle: 模型
        :param extend_words_vect: 扩展词 向量dict
        :param extend_words_index: 扩展词-index
        :param extend_words_matrix: 扩展词向量矩阵(按index排序)
        :param words_vect_matrix: (词向量+扩展词)向量矩阵 (按index排序)
        :param max_word_len: 词汇表大小
        '''

        self.vec_model_file = vec_model_file
        self.word_vect_dict = {}
        self.word_vect_modle = []
        self.extend_words_vect = {}
        self.extend_words_index = {}
        self.extend_words_matrix = []
        self.words_vect_matrix = []
        self.max_word_len = 0

    # 模型加载，初始化扩展词
    def load(self):
        self.word_vect_modle = Word2Vec.load_word2vec_format(self.vec_model_file, binary=False)
        print('load vector done!')
        self.max_word_len = len(self.word_vect_modle.wv.vocab)
        print('vocab_len:',self.max_word_len)
        self.init_data()
    def init_data(self):
        print('init words_vect_matrix')
        for index in range(self.max_word_len):
            word = self.word_vect_modle.index2word[index]
            self.words_vect_matrix.append(self.word_vect_modle[word])
        print('init words_vect_matrix done!',len(self.words_vect_matrix))
        init_extend_word_vec = [self.random_vect(word) for word in INIT_WORDS]
        print('init extend vec done!')

    #通过word 获取 index
    def get_int_by_word(self, word):
        if word in self.extend_words_index:
            return self.extend_words_index[word]
        if word in self.word_vect_modle.wv.vocab:
            return self.word_vect_modle.wv.vocab[word].index
        print(word, 'not in word_vec!')
        #return None
        return 1000

    # 通过 index 获取 word
    def get_word_by_int(self, index):
        if index < self.max_word_len:
            return self.word_vect_modle.index2word[index]
        return list(self.extend_words_index.keys())[list(self.extend_words_index.values()).index(index)]

    def get_vec_by_key(self, key):
        if key in self.word_vect_modle.wv.vocab:
            return self.word_vect_modle[key]
        elif key in self.extend_words_vect:
            return self.extend_words_vect[key]
        else:
            return self.random_vect(key)

    def random_vect(self, key):
        key_vect = Word2Vec.seeded_vector(self.word_vect_modle, key)
        print('random-',key)
        self.extend_words_vect[key] = key_vect
        self.extend_words_index[key] = self.max_word_len+ len(self.extend_words_index)
        self.extend_words_matrix.append(key_vect)
        self.words_vect_matrix.append(key_vect)
        return key_vect


if __name__ == '__main__':
    # model = my_word2vec('../model/finance/finance_jd_sg.txt')
    # model.load()
    if len(sys.argv) != 2:
        print('python word2vec/word2vec_tool.py  model_file')
    else:
        vec_model_file = sys.argv[1]  # 模型文件路径
        model = my_word2vec(vec_model_file)
        model.load()
        print(len(model.words_vect_matrix))
        # while True:
        #     input_str = raw_input('user>')
        #     if input_str == 'quit':
        #         exit()
        #     print 'input:', input_str
        #     if input_str.isdigit():
        #         word = model.get_word_by_int(int(input_str))
        #         print int(input_str),word
        #         continue
        #     word_vec = model.get_vec_by_key(input_str.decode('utf-8'))
        #     index = model.get_int_by_word(input_str.decode('utf-8'))
        #     print input_str, '：', word_vec
        #     print 'word:', word, 'index:', index
        #     for k, v in model.extend_words_index.items():
        #         print k, v
        #     print 'extend_words_matrix', model.extend_words_matrix
