
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

class DialogTrain:
    def __init__(self,checkpoint,source_file,target_file,\
                encoder_rnn_size,encoder_num_layers,decoder_rnn_size,decoder_num_layers,\
                epochs,batch_size,learn_rate,\
                word_to_vector,special_embeddings,general_embeddings
                ):
        self.checkpoint=checkpoint
        self.source_file=source_file
        self.target_file=target_file
        self.encoder_rnn_size=encoder_rnn_size
        self.encoder_num_layers=encoder_num_layers
        self.decoder_rnn_size=decoder_rnn_size
        self.decoder_num_layers=decoder_num_layers
        self.batch_size=batch_size
        self.epochs=epochs
        self.learn_rate=learn_rate
        self.special_embeddings=special_embeddings
        self.general_embeddings=general_embeddings
        self.word_to_vector=word_to_vector
        self.target_word_to_int={}
        self.target_int_to_word={}

    def test(self):
        print('是',self.word_to_vector.get_int_by_word('是'))
        print('的',self.word_to_vector.get_int_by_word('的'))

        print('没',self.word_to_vector.get_int_by_word('没'))
        print('问题',self.word_to_vector.get_int_by_word('问题'))
        print('什么',self.word_to_vector.get_int_by_word('什么'))
        print('时候',self.word_to_vector.get_int_by_word('时候'))
        print('还',self.word_to_vector.get_int_by_word('还'))
        #读取问和答数据，转成向量格式
        source_data,target_data=self.load_data(self.source_file,self.target_file) 
        print('source[0] ...')
        print(source_data[0])
        print(' target[0]...')
        print(target_data[0])

        self.target_int_to_word,self.target_word_to_int=self.extract_word_vocab(self.word_seg_sentences(target_data))
        print('target_int_to_word...')
        print(self.target_int_to_word)
        print('target_word_to_int...')
        print(self.target_word_to_int)

        source_pad_int=self.word_to_vector.get_int_by_word('<PAD>')
        target_eos_int=self.target_word_to_int['<EOS>']
        print('source pad int...')
        print(source_pad_int)
        print('target pad int...')
        print(target_eos_int)
        
        source_int = self.sentences_to_int(source_data,self.word_to_vector) 
        target_int = self.target_sentences_to_int(target_data,self.target_word_to_int) 
        print('source  int...')
        print(source_int)
        print('target int...')
        print(target_int)

        # 将数据集分割为train和validation
        train_source = source_int[self.batch_size:]
        train_target = target_int[self.batch_size:]
    
        print('train_source...')
        print(train_source)
        print('train_target...')
        print(train_target)
        # 留出一个batch进行验证
        valid_source = source_int[:self.batch_size]
        valid_target = target_int[:self.batch_size]
        print('valid_source...')
        print(valid_source)
        print('valid_target...')
        print(valid_target)

        (valid_targets_batch,valid_targets_limit_index_batch, valid_sources_batch,\
         valid_targets_lengths,valid_targets_limit_index_lengths, valid_sources_lengths) = \
            next(self.get_batches(valid_target, valid_source,valid_target_limit_index,\
            self.batch_size,source_pad_int,target_eos_int,target_eos_limit_index_int))
        print(valid_targets_batch, valid_sources_batch, valid_targets_lengths, valid_sources_lengths)

    def train(self):
        #读取问和答数据，转成向量格式
        source_data,target_data=self.load_data(self.source_file,self.target_file) 
        self.target_int_to_word,self.target_word_to_int=self.extract_word_vocab(self.word_seg_sentences(target_data))

        source_pad_int=self.word_to_vector.get_int_by_word('<PAD>')
        target_eos_int=self.word_to_vector.get_int_by_word('<EOS>')
        target_eos_limit_index_int=self.target_word_to_int['<EOS>']
        
        source_int = self.sentences_to_int(source_data,self.word_to_vector) 
        target_int = self.sentences_to_int(target_data,self.word_to_vector) 
        target_limit_index_int = self.target_limit_index_sentences_to_int(target_data,self.target_word_to_int) 

        # 将数据集分割为train和validation
        train_source = source_int[self.batch_size:]
        train_target = target_int[self.batch_size:]
        train_target_limit_index = target_limit_index_int[self.batch_size:]
    
        # 留出一个batch进行验证
        valid_source = source_int[:self.batch_size]
        valid_target = target_int[:self.batch_size]
        valid_target_limit_index = target_limit_index_int[self.batch_size:]
        (valid_targets_batch,valid_targets_limit_index_batch, valid_sources_batch, valid_targets_lengths,valid_targets_limit_index_lengths, valid_sources_lengths) = \
            next(self.get_batches(valid_target,valid_target_limit_index, valid_source, self.batch_size,source_pad_int,target_eos_int,target_eos_limit_index_int))
        
        display_step = 15 # 每隔50轮输出loss
    
        #train_graph,train_op,cost=self.create_graph(self.target_int_to_word,self.target_word_to_int,train_source)
        # 构造graph
        train_graph = tf.Graph()
        with train_graph.as_default():
            # 获得模型输入    
            targets,targets_limit_index,sources,\
            target_sequence_length,targets_limit_index_sequence_length, max_target_sequence_length, source_sequence_length,learn_rate = self.get_inputs()
            print('create seq2seq model...')
            training_decoder_output, predicting_decoder_output = self.seq2seq_model(\
                      sources,self.word_to_vector,source_sequence_length,\
                      targets,self.target_int_to_word,self.target_word_to_int,target_sequence_length,\
                      targets_limit_index,targets_limit_index_sequence_length,
                      max_target_sequence_length,\
                      self.encoder_rnn_size, self.encoder_num_layers,\
                      self.decoder_rnn_size, self.decoder_num_layers,\
                      self.special_embeddings,self.general_embeddings)
            
            training_logits = tf.identity(training_decoder_output.rnn_output, 'logits')
            training_ints = tf.identity(training_decoder_output.sample_id, 'logits_ints')
            predicting_result = tf.identity(predicting_decoder_output.rnn_output, name='predict')
            predicting_logits = tf.identity(predicting_decoder_output.sample_id, name='predictions')
            
            masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')
        
            print('create cost and optimization...')
            with tf.name_scope("optimization"):
                # Loss function
                cost = tf.contrib.seq2seq.sequence_loss(
                    #training_logits,
                    predicting_result,
                    targets_limit_index,
                    masks)
                # Optimizer
                optimizer = tf.train.AdamOptimizer(learn_rate)
                # Gradient Clipping
                gradients = optimizer.compute_gradients(cost)
                capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
                train_op = optimizer.apply_gradients(capped_gradients)
    
        print('now begin train...')
        with tf.Session(graph=train_graph) as sess:
            sess.run(tf.global_variables_initializer())
            for epoch_i in range(1, self.epochs+1):
                for batch_i, (targets_batch,targets_limit_index_batch, sources_batch, targets_lengths,targets_limit_index_lengths, sources_lengths) in enumerate(
                        self.get_batches(train_target,train_target_limit_index, train_source, self.batch_size,source_pad_int,target_eos_int,target_eos_limit_index_int)):
                    
                   # print('targets_batch',targets_batch)
                   # print('targets_limit_index_batch',targets_limit_index_batch)
                   # print('sources_batch',sources_batch)
                   # print('targets_lengths',targets_lengths)
                   # print('sources_lengths',sources_lengths)
                   # print('targets_limit_index_lengths',targets_limit_index_lengths)
                    _, loss = sess.run(
                        [train_op, cost],
                        {
                         targets: targets_batch,
                         targets_limit_index:targets_limit_index_batch,
                         sources: sources_batch,
                         learn_rate: self.learn_rate,
                         target_sequence_length: targets_lengths,
                         source_sequence_length: sources_lengths,
                         targets_limit_index_sequence_length:targets_limit_index_lengths})
        
                    if batch_i % display_step == 0:
                        # 计算validation loss
                        validation_loss = sess.run(
                        [cost],
                        {
                         targets: valid_targets_batch,
                         targets_limit_index:valid_targets_limit_index_batch,
                         sources: valid_sources_batch,
                         learn_rate: self.learn_rate,
                         target_sequence_length: valid_targets_lengths,
                         source_sequence_length: valid_sources_lengths,
                         targets_limit_index_sequence_length:valid_targets_limit_index_lengths})
                        
                        print('Epoch {:>3}/{} Batch {:>4}/{} - Training Loss: {:>6.3f}  - Validation loss: {:>6.3f}'
                              .format(epoch_i,
                                      self.epochs, 
                                      batch_i, 
                                      len(train_source) // self.batch_size, 
                                      loss, 
                                      validation_loss[0]))
                        answer_logit,train_ints=sess.run([predicting_logits,training_ints],
                         {sources: sources_batch,
                         targets: targets_batch,
                         target_sequence_length: targets_lengths,
                         source_sequence_length: sources_lengths,
                         targets_limit_index:targets_limit_index_batch,
                         learn_rate: self.learn_rate,
                         targets_limit_index_sequence_length:targets_limit_index_lengths
                         })
                        print('ask num:',len(sources_batch))
                        print('ask',[self.word_to_vector.get_word_by_int(item) for item in sources_batch[0]])
                        print('answer',[self.word_to_vector.get_word_by_int(item) for item in targets_batch[0]])
                        print('target',[self.target_int_to_word[item] for item in targets_limit_index_batch[0]])
                        print('answer int',targets_batch[0])
                        print('target int',targets_limit_index_batch[0])
                        print('trainn int',train_ints[0])
                        #print('train logit',train_logit[0])
                        print('predic int',answer_logit[0])
                        #print('should predict: {}'.format(" ".join([self.target_int_to_word[i] for i in answer_logit[0]])))
                        print('predict: {}'.format(" ".join([self.target_int_to_word[i] for i in answer_logit[0]])))
            
            # 保存模型
            saver = tf.train.Saver()
            print('Model begin save')
            saver.save(sess, self.checkpoint)
            print('Model Trained and Saved')
            
            #保存 target_word_to_int and target_int_to_word 
            print('save target_int_to_word to file ./target_int_to_word.json')
            with open('./target_int_to_word.json','w') as f:
                json.dump(self.target_int_to_word,f)
                print(self.target_int_to_word)
            print('save target_word_to_int to file ./target_word_to_int.json')
            with open('./target_word_to_int.json','w') as f2:
                json.dump(self.target_word_to_int,f2)
            
            #test predict begin...
            ask='最近比较困难'
            ask_int=self.sentence_to_int(ask,self.word_to_vector)
            input_data = train_graph.get_tensor_by_name('sources:0')
            logits_answer = train_graph.get_tensor_by_name('predictions:0')
            source_sequence_length = train_graph.get_tensor_by_name('source_sequence_length:0')
            target_sequence_length = train_graph.get_tensor_by_name('target_sequence_length:0')

            answer_logits = sess.run(logits_answer, {input_data: [ask_int]*self.batch_size,
                                              target_sequence_length: [20]*self.batch_size,
                                              source_sequence_length: [len(ask_int)]*self.batch_size})[0]
            print('\n原始输入:', ask)
            print('原始输入int:', ask_int)
            print(answer_logits)
            print('Target:')
            print('  Response Words: {}'.format(" ".join([self.target_int_to_word[i] for i in answer_logits])))
            #test predict end...
           
            return  self.target_int_to_word,self.target_word_to_int


    def pad_sentence_batch(self,sentence_batch, pad_int):
        '''
        对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length
        
        参数：
        - sentence batch
        - pad_int: <PAD>对应索引号
        '''
        max_sentence = max([len(sentence) for sentence in sentence_batch])+1
        return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]

    # ## Batches
    def get_batches(self,targets,targets_limit_index,sources, batch_size, source_pad_int, target_eos_int,target_eos_limit_index_int):
        '''
        定义生成器，用来获取batch大小的数据
        参数：
        - targets: int of targets.like:
              [[2,3],
               [5,4,5],
               ...,
              ]
        - sources: int of sources.like:
              [[1000,5230],
               [20,25,700],
               ...,
              ]
        - source_pad_vector: source 中<PAD>对应的int
        - target_pad_vector: target 中<PAD>对应的int
        '''
        for batch_i in range(0, len(sources)//batch_size):
            start_i = batch_i * batch_size
            sources_batch = sources[start_i:start_i + batch_size]
            targets_batch = targets[start_i:start_i + batch_size]
            targets_limit_index_batch = targets_limit_index[start_i:start_i + batch_size]
            # 补全序列
            #pad_sources_batch = np.array(self.pad_sentence_batch(sources_batch, source_pad_int))
            #pad_targets_batch = np.array(self.pad_sentence_batch(targets_batch, target_eos_int))
            pad_sources_batch = self.pad_sentence_batch(sources_batch, target_eos_int)
            pad_targets_batch = self.pad_sentence_batch(targets_batch, target_eos_int)
            pad_targets_limit_index_batch = self.pad_sentence_batch(targets_limit_index_batch, target_eos_limit_index_int)
            
            # 记录每条记录的长度
            targets_lengths = []
            for target in targets_batch:
                targets_lengths.append(len(target)+1)

            targets_limit_index_lengths = targets_lengths
            #for item in targets_limit_index_batch:
            #    targets_limit_index_lengths.append(len(target))
            
            source_lengths = []
            for source in sources_batch:
                source_lengths.append(len(source))
            
            yield pad_targets_batch,pad_targets_limit_index_batch,pad_sources_batch, targets_lengths,targets_limit_index_lengths, source_lengths

    # # 数据加载
    def load_data(self,source_file,target_file,encoding='utf-8'):
        """load data from source_file and target_file.
    
        - `source_file`:file name save ask sentences,format like: 
           你好
           今天天气怎么样
           ....
        - `target_file`:file name save answer sentences,format like: 
           你也好
           今天天气不错
           ....
        - `encoding`:encode of source_file and target_file,default utf8 
        -  return: list of source sentences and target sentences. like:
           (['你好','今天天气怎么样'...],['你也好','今天天气不错'...])
        """
        with open(source_file, 'r', encoding=encoding) as f:
            source_data = f.read().strip().split('\n')
        with open(target_file, 'r', encoding=encoding) as f:
            target_data = f.read().strip().split('\n')
        return source_data,target_data

    # # 生成词典
    def extract_word_vocab(self,data):
        '''
        构造映射表
        - `data`:list of list of word,format like:
           [['今天','天气','怎么样'],...]
    
        return: int_to_word and word_to_int.format like:
           ({5:'你好'...},{'天气':10})
        '''
        special_words = ['<PAD>', '<UNK>', '<GO>',  '<EOS>']
    
        set_words = list(set([word for line in data for word in line]))
        # 这里要把四个特殊字符添加进词典
        int_to_vocab = {int(idx): word for idx, word in enumerate(special_words + set_words)}
        vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}
    
        return int_to_vocab, vocab_to_int

    def word_seg_sentence(self,sentence):
        return list(jieba.cut(sentence.strip(),cut_all=False))
        
    def word_seg_sentences(self,data):
        result=[]
        for sentence in data:
            result.append(self.word_seg_sentence(sentence))
        return result

    # 上面已经构建完成Encoder和Decoder，下面将这两部分连接起来，构建seq2seq模型
    # In[13]:
    def seq2seq_model(self,encoder_input,word_to_vector,source_sequence_length,\
                      decoder_input,target_int_to_word,target_word_to_int,target_sequence_length,\
                      decoder_limit_index,targets_limit_index_sequence_length,
                      max_target_sequence_length,\
                      encoder_rnn_size, encoder_num_layers,\
                      decoder_rnn_size, decoder_num_layers,\
                      special_embeddings,general_embeddings):
        
        # 获取encoder的状态输出
        encoder_output, encoder_state = self.get_encoder_result(encoder_input,\
                           word_to_vector,\
                           source_sequence_length,\
                           encoder_rnn_size, encoder_num_layers)
        print('get encoder result...')
        
        #batch_size=len(encoder_input) 
        #batch_size=self.batch_size
        
        # 预处理后的decoder输入
        print('get decoder input...')
        decoder_input = self.process_decoder_input(decoder_input,word_to_vector, self.batch_size)
    
        # 将状态向量与输入传递给decoder
        training_decoder_output, predicting_decoder_output = self.get_decoder_result(decoder_input,\
                       target_int_to_word,target_word_to_int,target_sequence_length, max_target_sequence_length,\
                       word_to_vector,special_embeddings,\
                       decoder_num_layers, decoder_rnn_size,\
                       encoder_state,encoder_output,source_sequence_length,\
                       self.batch_size)
        return training_decoder_output, predicting_decoder_output
        print('get decoder result...')
    
    # ## Encoder
    # 在Encoder端，我们需要进行两步，第一步要对我们的输入进行Embedding，再把Embedding以后的向量传给RNN进行处理。
    def get_encoder_result(self,encoder_input,word_to_vector,\
                           source_sequence_length,
                           rnn_size, num_layers):
    
        '''
        构造Encoder层
        
        参数说明：
        - encoder_input: 输入单词id列表格式的句子。
        - special_source_int_to_word: 特殊字符的id:word 映射字典。
        - general_source_int_to_word: 通用字符的id:word 映射字典。
        - encoding_embedding_size: embedding的大小
        - source_sequence_length: 源数据的序列长度
        - rnn_size: rnn隐层结点数量
        - num_layers: 堆叠的rnn cell数量
        '''
        # Encoder embedding
       # print('type encoder_input')
       # print(type(encoder_input))
       # print('type special embedding')
       # print(type(self.special_embeddings))
       # print('type special embedding[0]')
       # print(type(self.special_embeddings[0]))
        #encoder_embeddings = tf.Variable(self.special_embeddings)
        encoder_embeddings = tf.constant(self.general_embeddings)
        encoder_embed_input = tf.nn.embedding_lookup(encoder_embeddings,encoder_input)
        print(encoder_input)
        print(type(encoder_embed_input))
        print(encoder_embed_input)
        #int_inputs_to_embed_inputs(encoder_input, source_int_to_word)
    
        # RNN cell
        def get_lstm_cell(rnn_size):
            lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            return lstm_cell
    
        cell = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(rnn_size) for _ in range(num_layers)])
        
        encoder_output, encoder_state = tf.nn.dynamic_rnn(cell, encoder_embed_input, 
                                                          sequence_length=source_sequence_length, dtype=tf.float32)
        
        return encoder_output, encoder_state

    
    # ### 对数据进行embedding
    # 同样地，我们还需要对target数据进行embedding，使得它们能够传入Decoder中的RNN。
    # Dense的说明在https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/layers/core.py
    def get_decoder_result(self,decoder_input,target_int_to_word,target_word_to_int,\
                       target_sequence_length, max_target_sequence_length,\
                       word_to_vector,special_embeddings,\
                       num_layers, rnn_size,\
                       encoder_state,encoder_output,source_sequence_length,\
                       batch_size):
        '''
        构造Decoder层
        
        参数：
        - decoder_input: list of list of word id,含有开始'<GO>'id,且去掉了最后一个id
        - target_int_to_word: target id:word 映射字典
        - target_sequence_length: target数据序列长度
        - max_target_sequence_length: target数据序列最大长度
        - word_to_int: <PAD>,<UNK>等特殊字符到id的映射表
        - special_embeddings: <PAD>,<UNK>等特殊字符的向量列表,按照id顺序排列
        - num_layers: 堆叠的RNN单元数量
        - rnn_size: RNN单元的隐层结点数量
        - encoder_state: encoder端编码的最终状态向量,大小等于decoder的rnn_size
        - encoder_output: encoder端编码的中间状态向量列表,长度等于encoder input列表长度， 每个向量大小等于decoder的rnn_size
        - source_sequence_length: encoder端输入的长度,决定attention层
        - batch_size: 一次训练的样本数量
        '''
        # 1. Embedding
        target_vocab_size = len(target_int_to_word)

        decoder_train_embeddings = tf.constant(self.general_embeddings)
        decoder_train_embed_input = tf.nn.embedding_lookup(decoder_train_embeddings,decoder_input)

        decoder_predict_embeddings = tf.constant(self.general_embeddings)
        #decoder_predict_embeddings =tf.Variable(tf.random_uniform([target_vocab_size, 100])) 
        decoder_predict_embed_input = tf.nn.embedding_lookup(decoder_predict_embeddings,decoder_input)
    
        # 2. 构造Decoder中的RNN单元
        def get_decoder_cell(rnn_size):
            decoder_cell = tf.contrib.rnn.LSTMCell(rnn_size,
                                               initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            return decoder_cell
        cell = tf.contrib.rnn.MultiRNNCell([get_decoder_cell(rnn_size) for _ in range(num_layers)])
    
        # 2.1 增加attention机制
        attension_menchian = tf.contrib.seq2seq.LuongAttention(rnn_size, encoder_output, source_sequence_length)
        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell, attension_menchian, rnn_size)
        decoder_initial_state = decoder_cell.zero_state(batch_size=batch_size, dtype=tf.float32).clone(cell_state=encoder_state)
    
        # 3. Output全连接层
        output_layer = Dense(target_vocab_size,
                             kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))
    
    
        # 4. Training decoder
        with tf.variable_scope("decode"):
            # 得到help对象
            training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_train_embed_input,
                                                                sequence_length=target_sequence_length,
                                                                time_major=False)
            # 构造decoder
            training_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                               training_helper,
                                                               decoder_initial_state,
                                                               output_layer) 
            training_decoder_output, _,_ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                           impute_finished=True,
                                                                           maximum_iterations=max_target_sequence_length)
        # 5. Predicting decoder
        # 与training共享参数
        with tf.variable_scope("decode", reuse=True):
            # 创建一个常量tensor并复制为batch_size的大小
            start_tokens = tf.tile(tf.constant([word_to_vector.get_int_by_word('<GO>')], dtype=tf.int32), [batch_size], 
                                   name='start_tokens')
            predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_predict_embeddings,
                                                                    start_tokens,
                                                                    word_to_vector.get_int_by_word('<EOS>'))
            predicting_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                            predicting_helper,
                                                            decoder_initial_state,
                                                            output_layer)
            predicting_decoder_output, _,_ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder,
                                                                impute_finished=True,
                                                                maximum_iterations=max_target_sequence_length)
        
        return training_decoder_output, predicting_decoder_output
    # ## Decoder
    # ### 对target数据进行预处理
    def process_decoder_input(self,data,word_to_vector, batch_size):
        '''
        补充<GO>，并移除最后一个字符
        '''
        # cut掉最后一个字符
        ending = tf.strided_slice(data, [0, 0], [batch_size, -1], [1, 1])
        decoder_input = tf.concat([tf.fill([batch_size, 1], word_to_vector.get_int_by_word('<GO>')), ending], 1)
        #ending = tf.strided_slice(data, [0, 0], [batch_size, -1], [1, 1])
        #decoder_input = tf.concat([tf.fill([batch_size, 1], word_to_vector.get_int_by_word('<GO>')), data], 1)
        return decoder_input
    
    def create_graph(self,target_int_to_word,target_word_to_int,train_source):
        # 构造graph
        train_graph = tf.Graph()
        
        with train_graph.as_default():
            
            # 获得模型输入    
            input_data, targets,targets_limit_index, learn_rate, target_sequence_length, max_target_sequence_length, source_sequence_length = self.get_inputs()
            
            training_decoder_output, predicting_decoder_output = self.seq2seq_model(\
                      input_data,self.word_to_vector,source_sequence_length,\
                      targets,target_int_to_word,target_word_to_int,target_sequence_length,\
                      max_target_sequence_length,\
                      self.encoder_rnn_size, self.encoder_num_layers,\
                      self.decoder_rnn_size, self.decoder_num_layers,\
                      self.special_embeddings,self.general_embeddings)
            
            training_logits = tf.identity(training_decoder_output.rnn_output, 'logits')
            predicting_logits = tf.identity(predicting_decoder_output.sample_id, name='predictions')
            
            masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')
        
            with tf.name_scope("optimization"):
                # Loss function
                cost = tf.contrib.seq2seq.sequence_loss(
                    training_logits,
                    targets_limit_index,
                    masks)
                # Optimizer
                optimizer = tf.train.AdamOptimizer(learn_rate)
                # Gradient Clipping
                gradients = optimizer.compute_gradients(cost)
                capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
                train_op = optimizer.apply_gradients(capped_gradients)
        return train_graph,train_op,cost

    def sentence_to_int(self,sentence,word_to_vector):
        return [word_to_vector.get_int_by_word(word) for word in self.word_seg_sentence(sentence)]
    
    def sentences_to_int(self,sentences,word_to_vector):
        return [self.sentence_to_int(sentence,word_to_vector) for sentence in sentences]

    def target_limit_index_sentence_to_int(self,sentence,target_word_to_int):
        return [target_word_to_int.get(word, target_word_to_int['<UNK>']) for word in self.word_seg_sentence(sentence)]
    
    def target_limit_index_sentences_to_int(self,sentences,target_word_to_int):
        return [self.target_limit_index_sentence_to_int(sentence,target_word_to_int) for sentence in sentences]

    def get_inputs(self):
        '''
        模型输入tensor
        '''
        sources = tf.placeholder(tf.int32, [None, None], name='sources')
        targets = tf.placeholder(tf.int32, [None, None], name='targets')
        targets_limit_index = tf.placeholder(tf.int32, [None, None], name='targets_limit_index')
        learn_rate = tf.placeholder(tf.float32, name='learn_rate')
        
        # 定义target序列最大长度（之后target_sequence_length和source_sequence_length会作为feed_dict的参数）
        target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
        target_limit_index_sequence_length = tf.placeholder(tf.int32, (None,), name='target_limit_index_sequence_length')
        max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len')
        source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')
        
        return  targets,targets_limit_index,sources,target_sequence_length,target_limit_index_sequence_length,\
             max_target_sequence_length, source_sequence_length,learn_rate 

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
            if item==None:
                print(item)
                item=1000
            embed_input.append(word_to_vector.get_vec_by_key(word_to_vector.get_word_by_int(item)))
        return embed_input
    
    def source_int_inputs_to_embed_inputs(self,int_inputs,word_to_vector):
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
            embed_inputs.append(self.source_int_input_to_embed_input(item,word_to_vector))
        return embed_inputs
    
    # convert int input to embed input
    def target_int_input_to_embed_input(self,int_input,word_to_vector):
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
            embed_input.append(word_to_vector(self.target_int_to_word.get(item,'<UNK>')))
        return embed_input
    
    def target_int_inputs_to_embed_inputs(self,int_inputs,word_to_vector):
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
            embed_inputs.append(self.int_input_to_embed_input(item,word_to_vector))
        return embed_inputs

    def get_target_word_to_int(self):
        return self.target_word_to_int
    def get_target_int_to_word(self):
        return self.target_int_to_word
