
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

class DialogTrain:
    def __init__(self,checkpoint = "./dialog_model.ckpt",source_file,target_file,\
                special_word_to_int,special_int_to_word,special_embeddings,\
                general_word_to_int,general_int_to_word,\
                word_to_vector,\
                encoder_rnn_size,encoder_num_layers,decoder_rnn_size,decoder_num_layers,\
                epochs,batch_size,learn_rate):
        self.checkpoint=checkpoint
        self.source_file=source_file
        self.target_file=target_file
        self.target_file=target_file
        self.special_word_to_int=special_word_to_int
        self.special_int_to_word=special_int_to_word
        self.special_embeddings=special_embeddings
        self.general_word_to_int=general_word_to_int
        self.general_int_to_word=general_int_to_word
        self.word_to_vector=word_to_vector
        self.encoder_rnn_size=encoder_rnn_size
        self.encoder_num_layers=encoder_num_layers
        self.decoder_rnn_size=decoder_rnn_size
        self.decoder_num_layers=decoder_num_layers
        self.batch_size=batch_size
        self.epochs=epochs
        self.learn_rate=learn_rate

    def train(self):
        #读取问和答数据，转成向量格式
        source_data,target_data=load_data(self.source_file,self.target_file) 
        target_word_to_int,target_int_to_word=extract_word_vocab(word_seg_sentences(target_data))

        source_pad_int=self.special_word_to_int['<PAD>']
        target_pad_int=target_word_to_int['<PAD>']
        
        source_int = source_sentences_to_int(source_data,self.special_word_to_int,self.general_word_to_int) 
        target_int = target_sentences_to_int(target_data,target_word_to_int) 
    
        # 将数据集分割为train和validation
        train_source = source_int[self.batch_size:]
        train_target = target_int[self.batch_size:]
    
        # 留出一个batch进行验证
        valid_source = source_int[:self.batch_size]
        valid_target = target_int[:self.batch_size]
        (valid_targets_batch, valid_sources_batch, valid_targets_lengths, valid_sources_lengths) = \
            next(get_batches(valid_target, valid_source, batch_size,source_pad_int,target_pad_int))
        
        display_step = 50 # 每隔50轮输出loss
    
        train_graph,train_op,cost=create_graph(source_int_to_word,target_int_to_word,\
                     special_word_to_int,special_embeddings,rnn_size, num_layers):
    
        with tf.Session(graph=train_graph) as sess:
            sess.run(tf.global_variables_initializer())
                
            for epoch_i in range(1, epochs+1):
                for batch_i, (targets_batch, sources_batch, targets_lengths, sources_lengths) in enumerate(
                        get_batches(train_target, train_source, batch_size,source_pad_int,target_pad_vector)):
                    
                    _, loss = sess.run(
                        [train_op, cost],
                        {input_data: sources_batch,
                         targets: targets_batch,
                         lr: learning_rate,
                         target_sequence_length: targets_lengths,
                         source_sequence_length: sources_lengths})
        
                    if batch_i % display_step == 0:
                        # 计算validation loss
                        validation_loss = sess.run(
                        [cost],
                        {input_data: valid_sources_batch,
                         targets: valid_targets_batch,
                         lr: learning_rate,
                         target_sequence_length: valid_targets_lengths,
                         source_sequence_length: valid_sources_lengths})
                        
                        print('Epoch {:>3}/{} Batch {:>4}/{} - Training Loss: {:>6.3f}  - Validation loss: {:>6.3f}'
                              .format(epoch_i,
                                      epochs, 
                                      batch_i, 
                                      len(train_source) // batch_size, 
                                      loss, 
                                      validation_loss[0]))
            
            # 保存模型
            saver = tf.train.Saver()
            print('Model begin save')
            saver.save(sess, checkpoint)
            print('Model Trained and Saved')


    def get_batches(targets, sources, batch_size, source_pad_int, target_pad_int):
        '''
        定义生成器，用来获取batch
        参数：
        - targets: int of targets.like:
              [[1,2,3],
               [5,4,5],
               ...,
              ]
        - sources: int of sources.like:
              [[1,5,3],
               [2,2,7],
               ...,
              ]
        - source_pad_vector: <PAD>对应的int
        - target_pad_vector: <PAD>对应的int
        '''
        for batch_i in range(0, len(sources)//batch_size):
            start_i = batch_i * batch_size
            sources_batch = sources[start_i:start_i + batch_size]
            targets_batch = targets[start_i:start_i + batch_size]
            # 补全序列
            pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
            pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))
            
            # 记录每条记录的长度
            targets_lengths = []
            for target in targets_batch:
                targets_lengths.append(len(target))
            
            source_lengths = []
            for source in sources_batch:
                source_lengths.append(len(source))
            
            yield pad_targets_batch, pad_sources_batch, targets_lengths, source_lengths
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
            source_data = f.read().split('\n')
        with open(target_file, 'r', encoding=encoding) as f:
            target_data = f.read().split('\n')
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
        int_to_vocab = {idx: word for idx, word in enumerate(special_words + set_words)}
        vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}
    
        return int_to_vocab, vocab_to_int

    def word_seg_sentence(self,sentence):
        return list(jieba.cut(sentence.strip(),cut_all=False))
        
    def word_seg_sentences(self,data):
        result=[]
        for sentence in data:
            result.append(word_seg_sentence(sentence))


def target_sentence_to_int(sentence,word_to_int):
    return [word_to_int.get(word, word_to_int['<UNK>']) for word in word_seg_sentence(sentence)]

def target_sentences_to_int(sentences,word_to_int):
    return [sentence_to_int(sentence) for sentence in sentences]

def source_sentence_to_int(sentence,special_word_to_int,general_word_to_int):
    word_list=word_seg_sentence(sentence)
    result=[]
    for word in word_list:
        if word in special_word_to_int:
            result.append(special_word_to_int.get(word)
        elif word in general_word_to_int:
            result.append(general_word_to_int.get(word)
        else:
            result.append(special_word_to_int.get('<UNK>')
    return result

def source_sentences_to_int(sentences,special_word_to_int,general_word_to_int):
    return [source_sentence_to_int(sentence,special_word_to_int,general_word_to_int) for sentence in sentences]

#def source_data_to_int(source_data,special_word_to_int,general_word_to_int)
#    '''
#    - `source_data`:list of list of word,format like:
#       [['今天','天气','怎么样'],...]
#
#    return: list of list of int,format like:
#       [[6,10,12],...]
#    '''
#    result=[]
#    for word in source_data:
#        if word in special_word_to_int:
#            result.append(special_word_to_int.get(word)
#        elif word in general_word_to_int:
#            result.append(general_word_to_int.get(word)
#        else:
#            result.append(special_word_to_int.get('<UNK>')
#    return result
#
#def source_data_to_int(source_data,special_word_to_int,general_word_to_int)
#    '''
#    - `source_data`:list of list of word,format like:
#       [['今天','天气','怎么样'],...]
#
#    return: list of list of int,format like:
#       [[6,10,12],...]
#    '''
#    result=[]
#    for word in source_data:
#        if word in special_word_to_int:
#            result.append(special_word_to_int.get(word)
#        elif word in general_word_to_int:
#            result.append(general_word_to_int.get(word)
#        else:
#            result.append(special_word_to_int.get('<UNK>')
#    return result

# 构造映射表
#source_int_to_word, source_word_to_int = extract_word_vocab(source_data)
#target_int_to_word, special_word_to_int = extract_word_vocab(target_data)

# 对字母进行转换

# # 构建模型

# ## 输入层

def get_inputs():
    '''
    模型输入tensor
    '''
    inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    
    # 定义target序列最大长度（之后target_sequence_length和source_sequence_length会作为feed_dict的参数）
    target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
    max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len')
    source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')
    
    return inputs, targets, learning_rate, target_sequence_length, max_target_sequence_length, source_sequence_length

# convert int input to embed input
def source_int_input_to_embed_input(int_input,special_int_to_word,general_int_to_word):
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
        if item in special_int_to_word:
            word=special_int_to_word.get(item)
        elif item in general_int_to_word:
            word=general_int_to_word.get(item)
        else:
            word='<UNK>'
        embed_input.append(word_to_vector(word))
    return embed_input

def source_int_inputs_to_embed_inputs(int_inputs,special_int_to_word,general_int_to_word):
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
        embed_inputs.append(source_int_input_to_embed_input(item,special_int_to_word,general_int_to_word))
    return embed_inputs

# convert int input to embed input
def target_int_input_to_embed_input(int_input,int_to_word):
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
        embed_input.append(word_to_vector(int_to_word[item]))
    return embed_input

def targets_int_inputs_to_embed_inputs(int_inputs,int_to_word):
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
        embed_inputs.append(int_input_to_embed_input(item,int_to_word))
    return embed_inputs

# ## Encoder

# 在Encoder端，我们需要进行两步，第一步要对我们的输入进行Embedding，再把Embedding以后的向量传给RNN进行处理。
# 
# 在Embedding中，我们使用[tf.contrib.layers.embed_sequence](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/embed_sequence)，它会对每个batch执行embedding操作。

# In[10]:


def get_encoder_result(encoder_input,source_int_to_word,\
                       source_sequence_length,
                       rnn_size, num_layers):

    '''
    构造Encoder层
    
    参数说明：
    - encoder_input: 输入单词id列表格式的句子。
    - source_int_to_word: id:word 映射字典。
    - encoding_embedding_size: embedding的大小
    - source_sequence_length: 源数据的序列长度
    - rnn_size: rnn隐层结点数量
    - num_layers: 堆叠的rnn cell数量
    '''
    # Encoder embedding
    encoder_embed_input = int_inputs_to_embed_inputs(encoder_input, source_int_to_word)

    # RNN cell
    def get_lstm_cell(rnn_size):
        lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return lstm_cell

    cell = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(rnn_size) for _ in range(num_layers)])
    
    encoder_output, encoder_state = tf.nn.dynamic_rnn(cell, encoder_embed_input, 
                                                      sequence_length=source_sequence_length, dtype=tf.float32)
    
    return encoder_output, encoder_state


# ## Decoder

# ### 对target数据进行预处理

# In[11]:


def process_decoder_input(data, vocab_to_int, batch_size):
    '''
    补充<GO>，并移除最后一个字符
    '''
    # cut掉最后一个字符
    ending = tf.strided_slice(data, [0, 0], [batch_size, -1], [1, 1])
    decoder_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)

    return decoder_input


# ### 对数据进行embedding
# 
# 同样地，我们还需要对target数据进行embedding，使得它们能够传入Decoder中的RNN。
# 
# Dense的说明在https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/layers/core.py

# In[12]:


def get_decoder_result(decoder_input,target_int_to_word,\
                   target_sequence_length, max_target_sequence_length,\
                   special_word_to_int,special_embeddings,\
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
    - special_word_to_int: <PAD>,<UNK>等特殊字符到id的映射表
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
    #decoder_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    decoder_embed_input = int_inputs_to_embed_inputs(decoder_input,target_int_to_word)

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
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input,
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
        start_tokens = tf.tile(tf.constant([special_word_to_int['<GO>']], dtype=tf.int32), [batch_size], 
                               name='start_tokens')
        predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(special_embeddings,
                                                                start_tokens,
                                                                special_word_to_int['<EOS>'])
        predicting_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                        predicting_helper,
                                                        decoder_initial_state,
                                                        output_layer)
        predicting_decoder_output, _,_ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder,
                                                            impute_finished=True,
                                                            maximum_iterations=max_target_sequence_length)
    
    return training_decoder_output, predicting_decoder_output


# ### Seq2Seq
# 
# 上面已经构建完成Encoder和Decoder，下面将这两部分连接起来，构建seq2seq模型

# In[13]:


def seq2seq_model(encoder_input,source_int_to_word,source_sequence_length,\
                  decoder_input,target_int_to_word,target_sequence_length,\
                  max_target_sequence_length,\
                  rnn_size, num_layers,\
                  special_word_to_int,special_embeddings):
    
    # 获取encoder的状态输出
    encoder_output, encoder_state = get_encoder_result(encoder_input,\
                       source_int_to_word,\
                       source_sequence_length,\
                       rnn_size, num_layers)
    
    batch_size=len(encoder_input) 
    
    # 预处理后的decoder输入
    decoder_input = process_decoder_input(decoder_input, special_word_to_int, batch_size)


    # 将状态向量与输入传递给decoder
    training_decoder_output, predicting_decoder_output = get_decoder_result(decoder_input,\
                   target_int_to_word,target_sequence_length, max_target_sequence_length,\
                   special_word_to_int,special_embeddings,\
                   num_layers, rnn_size,\
                   encoder_state,encoder_output,source_sequence_length,\
                   batch_size):
    return training_decoder_output, predicting_decoder_output

# In[15]:

def create_graph(source_int_to_word,target_int_to_word,\
                 special_word_to_int,special_embeddings,rnn_size, num_layers):
    # 构造graph
    train_graph = tf.Graph()
    
    with train_graph.as_default():
        
        # 获得模型输入    
        input_data, targets, learn_rate, target_sequence_length, max_target_sequence_length, source_sequence_length = get_inputs()
        
        training_decoder_output, predicting_decoder_output = seq2seq_model(\
                  input_data,source_int_to_word,source_sequence_length,\
                  targets,target_int_to_word,target_sequence_length,\
                  max_target_sequence_length,\
                  rnn_size, num_layers,\
                  special_word_to_int,special_embeddings):
        
        training_logits = tf.identity(training_decoder_output.rnn_output, 'logits')
        predicting_logits = tf.identity(predicting_decoder_output.sample_id, name='predictions')
        
        masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')
    
        with tf.name_scope("optimization"):
            
            # Loss function
            cost = tf.contrib.seq2seq.sequence_loss(
                training_logits,
                targets,
                masks)
    
            # Optimizer
            optimizer = tf.train.AdamOptimizer(learn_rate)
    
            # Gradient Clipping
            gradients = optimizer.compute_gradients(cost)
            capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
            train_op = optimizer.apply_gradients(capped_gradients)

    return train_graph,train_op,cost

# ## Batches

# In[16]:


def pad_sentence_batch(sentence_batch, pad_int):
    '''
    对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length
    
    参数：
    - sentence batch
    - pad_int: <PAD>对应索引号
    '''
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]


# In[17]:




# ## Train

# In[18]:
def get_special_vector(char):
    return [1,2,3]

def word_to_vector(word):
    return [1,2,3]

def sentence_to_vector(sentence):
    result=[]
    word_list=word_seg_sentence(sentence)
    for word in word_list:
        result.append(word_to_vector(word))
    return result

def sentences_to_vector(sentences):
    result=[]
    for sentence in sentences:
        result.append(sentence_to_vector(sentence))
    return result



# ## 预测

# In[21]:


#def source_to_seq(text):
#    '''
#    对源数据进行转换
#    '''
#    sequence_length = 7
#    return [source_word_to_int.get(word, source_word_to_int['<UNK>']) for word in text] + [source_word_to_int['<PAD>']]*(sequence_length-len(text))


# In[24]:
def predict(checkpoint,ask):

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

