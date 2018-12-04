# coding:utf-8
from __future__ import print_function
import sys
import numpy as np
from numpy import zeros
from gensim.models.word2vec import Word2Vec
from gensim import utils

# 1.扩展 words_index ={word:index}
# 2.扩展 矩阵 按 index 排序
# 3.get vect  by word
# ** 使用时只用到了 初始化词的 索引值
INIT_WORDS = ['<PAD>', '<UNK>', '<GO>', '<EOS>']
SPACE=' '

class MyWord2vec(object):
    def __init__(self, vec_model_file,extend_vect_file):
        '''
        :param vec_model_file: 模型文件路径
        :param word_vect_dict: 词向量 dict
        :param word_vect_modle: 模型
        :param extend_words_vect: 扩展词 向量dict
        :param extend_words_index: 扩展词-index
        :param extend_index_words: index-扩展词
        :param extend_words_vect_matrix: 扩展词向量矩阵(按index排序)
        :param words_vect_matrix: (词向量+扩展词)向量矩阵 (按index排序)
        :param max_word_len: 词汇表大小
        '''

        self.vec_model_file = vec_model_file
        self.extend_vect_file = extend_vect_file
        self.word_vect_dict = {}
        self.word_vect_modle = []
        self.extend_word_vect_modle = []
        self.extend_words_vect = {}
        self.extend_words_index = {}
        self.extend_index_words = {}
        self.extend_words_vect_matrix = []
        self.words_vect_matrix = []
        self.max_word_len = 0

    # 模型加载，初始化扩展词
    def load(self):
        self.word_vect_modle = Word2Vec.load_word2vec_format(self.vec_model_file, binary=False)
        print('load word2vector model done!')
        self.max_word_len = len(self.word_vect_modle.wv.vocab)
        print('all word num is:',self.max_word_len)
        self.init_data()
    def init_data(self):
        print('init words_vect_matrix')
        for index in range(self.max_word_len):
            word = self.word_vect_modle.index2word[index]
            self.words_vect_matrix.append(self.word_vect_modle[word])
        print('init words_vect_matrix done!',len(self.words_vect_matrix))
        # init_extend_word_vec = [self.random_vect(word) for word in INIT_WORDS]
        #print('init extend vec done!')
        self.read_extend_file()

    #通过word 获取 index
    def get_int_by_word(self, word,default_value=482842):
        if word in self.extend_words_index:
            return self.extend_words_index[word]
        if word in self.word_vect_modle.wv.vocab:
            return self.word_vect_modle.wv.vocab[word].index
        print(word, 'not in word_vec, begin create it!')
       # self.random_vect(word)
       # if word in self.extend_words_index:
       #     return self.extend_words_index[word]
        return default_value

    # 通过 index 获取 word
    def get_word_by_int(self, index):
        if index < self.max_word_len:
            return self.word_vect_modle.index2word[index]
        return self.extend_index_words[index]

    def get_vec_by_word(self, word):
        if word in self.word_vect_modle.wv.vocab:
            return self.word_vect_modle[word]
        elif word in self.extend_words_vect:
            return self.extend_words_vect[word]
        else:
            return self.random_vect(word)

    def random_vect(self, word):
        word_vect = Word2Vec.seeded_vector(self.word_vect_modle, word)
        print('random-',word)
        print('type random word_vect',type(word_vect))
        self.extend_words_vect[word] = word_vect
        index = self.max_word_len+ len(self.extend_words_index)
        self.extend_words_index[word] = index
        self.extend_index_words[index] = word
        self.extend_words_vect_matrix.append(word_vect)
        self.words_vect_matrix.append(word_vect)
        self.write_extend_file(word,word_vect)
        return word_vect

    def read_extend_file(self):
        print('now read extend file')
        self.extend_word_vect_modle = Word2Vec.load_word2vec_format(self.extend_vect_file, binary=False)
        max_extend_word_len = len(self.extend_word_vect_modle.wv.vocab)
        for index in range(max_extend_word_len):
            word = self.extend_word_vect_modle.index2word[index]
            self.words_vect_matrix.append(self.extend_word_vect_modle[word])
            print('type self.extend_word_vect_modle[word] ',type(self.extend_word_vect_modle[word]))
            self.extend_words_vect_matrix.append(self.extend_word_vect_modle[word])
            self.extend_words_vect[word] = self.extend_word_vect_modle[word].tolist()
            self.extend_words_index[word] = self.max_word_len+index
            self.extend_index_words[self.max_word_len+index] = word

       # inf = open(self.extend_vect_file, 'r')
       # print('now read extend file')
       # for index, line in enumerate(inf.readlines()):
       #     word_vect = line.strip().split(' ')
       #     word = word_vect[0]
       #     vects = word_vect[1:]
       #     vects_f=[float(item) for item in vects]
       #     self.extend_words_vect[word] = vects_f
       #     self.extend_words_vect_matrix.append(np.array(vects_f))
       #     self.extend_words_index[word] = self.max_word_len+index
       #     self.extend_index_words[self.max_word_len+index] = word
       #     self.words_vect_matrix.append(np.array(vects_f))
       #     print('word:%s,index:%s'%(word,str(self.max_word_len+index)))
       # inf.close()
        print('now read extend file done!')

    def write_extend_file(self,word,vects):
        print('write extend vects')
        print(word,vects)
        inf = open(self.extend_vect_file, 'a')
        #print('write------------',word)
        print('now append vect file',word)
        print(vects)
        inf.write((str)(utils.to_utf8("%s %s\n" % (word, SPACE.join("%f" % val for val in vects)))))
        inf.close()
        print('now update extend vect file total line num')
        with open(self.extend_vect_file, 'r') as f_read:
            lines=f_read.readlines()
        f_read.close()
        with open(self.extend_vect_file, 'w') as f_write:
            line_0=lines[0]
            line_0_split=line_0.strip().split(' ')
            line_0_split[0]=str(((int)(line_0_split[0]))+1)
            line_0_new=line_0_split[0]+' '+line_0_split[1]+'\n'
            lines[0]=line_0_new
            for line in lines:
                f_write.write(line)
        f_write.close()
        


if __name__ == '__main__':
    # model = my_word2vec('../model/finance/finance_jd_sg.txt')
    # model.load()
    if len(sys.argv) != 3:
        print('python word2vec/word2vec_tool.py  model_file')
    else:
        vec_model_file = sys.argv[1]  # 模型文件路径
        extend_vect_file = sys.argv[2]
        model = my_word2vec(vec_model_file,extend_vect_file)
        model.load()
        while True:
            input_str = raw_input('user>')
            if input_str == 'quit':
                exit()
            if input_str.isdigit():
                word = model.get_word_by_int(int(input_str))
                print(int(input_str),word,model.get_vec_by_word(word))
                continue
            word_vec = model.get_vec_by_word(input_str.decode('utf-8'))
            index = model.get_int_by_word(input_str.decode('utf-8'))
            print(input_str, ':', word_vec,index)
            for k, v in model.extend_words_index.items():
                print(k, v)
        print('extend_words_vect_matrix', model.extend_words_vect_matrix)
