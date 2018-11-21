
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

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.8'), 'Please use TensorFlow version 1.8 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

import numpy as np
import time
import tensorflow as tf
import jieba

class DialogPredict:
    def __init__(self,checkpoint = "./dialog_model.ckpt",target_int_to_word,\
        word_to_int,int_to_word,\
        word_to_vector):
        self.checkpoint=checkpoint
        self.target_int_to_word=target_int_to_word
        self.word_to_int=word_to_int
        self.general_word_to_int=general_word_to_int
        self.int_to_word=int_to_word
        self.general_int_to_word=general_int_to_word
        self.word_to_vector=word_to_vector

    def source_sentence_to_int(self,sentence,):
        word_list=self.word_seg_sentence(sentence)
        result=[]
        for word in word_list:
            if word in word_to_int:
                result.append(self.word_to_int.get(word)
            elif word in general_word_to_int:
                result.append(self.general_word_to_int.get(word)
            else:
                result.append(self.word_to_int.get('<UNK>')

    def word_seg_sentences(self,data):
        result=[]
        for sentence in data:
            result.append(word_seg_sentence(sentence))

    # convert int input to embed input
    def source_int_input_to_embed_input(self,int_input,int_to_word,general_int_to_word):
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
            word='<UNK>'
            if item in int_to_word:
                word=int_to_word.get(item)
            elif item in general_int_to_word:
                word=general_int_to_word.get(item)
            else:
                word='<UNK>'
            embed_input.append(self.word_to_vector(word))
        return embed_input

    def source_int_inputs_to_embed_inputs(self,int_inputs):
        '''
        参数说明：
        - int_inputs: sentences.list of list of 输入单词id的列表,format like:
            [[1,15,3,4],[2,6,7,8]]
        - int_to_word: 字典映射of id:word
        返回值: 
        - embed_inputs:sentences,list of list of vector,formate like:
            [[[1.0,2.1,3.5],[2.2,5.9,1.3],[2.2,3.2,2.1],[1.5,6.3.8.0]],[[1.0,2.1,3.5],[2.2,5.9,1.3],[2.2,3.2,2.1],[1.5,6.3.8.0]]]
        '''
        embed_inputs=[]
        for item in int_inputs:
            embed_inputs.append(self.source_int_input_to_embed_input(item,int_to_word,general_int_to_word))
        return embed_inputs

    def predict(self,checkpoint,ask):
        ask_int = sentence_to_int(ask)
        loaded_graph = tf.Graph()
        with tf.Session(graph=loaded_graph) as sess:
            # 加载模型
            loader = tf.train.import_meta_graph(checkpoint + '.meta')
            loader.restore(sess, checkpoint)
        
            input_data = loaded_graph.get_tensor_by_name('inputs:0')
            logits = loaded_graph.get_tensor_by_name('predictions:0')
            source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_length:0')
            target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')
            
            answer_logits = sess.run(logits, {input_data: [ask_int]*batch_size, 
                                              target_sequence_length: [target_sequence_length]*batch_size, 
                                              source_sequence_length: [len(ask)]*batch_size})[0] 
        
        
        pad = source_word_to_int["<PAD>"] 
        
        print('原始输入:', ask)
        
        print('\nTarget')
        print('  Response Words: {}'.format(" ".join([target_int_to_word[i] for i in answer_logits if i != pad])))

