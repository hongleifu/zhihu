
# coding: utf-8

# # Seq2Seq
# 
# 本篇代码将实现一个基础版的Seq2Seq，输入一个单词（字母序列），模型将返回一个对字母排序后的“单词”。
# 
# 基础Seq2Seq主要包含三部分：
# 
# - Encoder
# - 隐层状态向量（连接Encoder和Decoder）
# - Decoder

# # 查看TensorFlow版本

# In[1]:


from distutils.version import LooseVersion
import tensorflow as tf
from tensorflow.python.layers.core import Dense
import json

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.8'), 'Please use TensorFlow version 1.8 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

import numpy as np
import time
import tensorflow as tf
import jieba

class DialogPredict:
    def __init__(self,checkpoint,target_int_to_word_file,target_word_to_int_file,\
        word_to_vector,batch_size):
        #初始化参数
        self.checkpoint=checkpoint
        with open(target_int_to_word_file,'r') as f:
            self.target_int_to_word=json.load(f)
        with open(target_word_to_int_file,'r') as f:
            self.target_word_to_int=json.load(f)
        self.word_to_vector=word_to_vector
        self.batch_size=batch_size

        # 加载模型
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess=tf.Session(graph=self.graph)
        with self.sess.as_default():
            with self.graph.as_default():
                loader = tf.train.import_meta_graph(self.checkpoint + '.meta')
                loader.restore(self.sess, self.checkpoint)

    def source_sentence_to_int(self,sentence,word_to_vector):
        return [word_to_vector.get_int_by_word(word) for word in self.word_seg_sentence(sentence)]+[word_to_vector.get_int_by_word('<EOS>')]
        #return [word_to_vector.get_int_by_word(word) for word in self.word_seg_sentence(sentence)]

    def word_seg_sentence(self,sentence):
        return list(jieba.cut(sentence.strip(),cut_all=False))

    # convert int input to embed input
    def source_int_input_to_embed_input(self,int_input,word_to_vector):
        '''
        参数说明：
        - int_input: one sentence,a list of 输入单词id的列表,format like:
            [1,15,3,4]
        - int_to_word: 字典映射of id:word
        返回值: 
        - embed_input,one sentence,a list of vector,formate like:
            [[1.0,2.1,3.5],[2.2,5.9,1.3],[2.2,3.2,2.1],[1.5,6.3.8.0]]
        '''
        embed_input=[]
        for item in int_input:
            embed_input.append(word_to_vector.get_vec_by_word(word_to_vector.get_word_by_int(item)))
        return embed_input
    
    def predict(self,ask):
        ask_int = self.source_sentence_to_int(ask,self.word_to_vector)
        ask_vector = self.source_int_input_to_embed_input(ask_int,self.word_to_vector)

        input_data = self.graph.get_tensor_by_name('sources:0')
        logits = self.graph.get_tensor_by_name('predictions:0')
        source_sequence_length = self.graph.get_tensor_by_name('source_sequence_length:0')
        target_sequence_length = self.graph.get_tensor_by_name('target_sequence_length:0')

        answer_logits = self.sess.run(logits, {input_data: [ask_int]*self.batch_size, 
                                          target_sequence_length: [50]*self.batch_size, 
                                          source_sequence_length: [len(ask_int)]*self.batch_size})[0] 
        
        
        eos = self.target_word_to_int["<EOS>"] 
        #print(self.target_int_to_word) 
        
        print('\n原始输入:', ask)
        print('原始输入int:', ask_int)
        print('原始输入分词:', self.word_seg_sentence(ask))
        
        print(answer_logits)
        print('Target:')
        print('  Response Words: {}'.format(" ".join([self.target_int_to_word[str(i)] for i in answer_logits])))

