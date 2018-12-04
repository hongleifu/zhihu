# coding: utf-8
from __future__ import print_function
import Seq2seq_dialog_train
import Seq2seq_dialog_predict
from word2vect import word2vec_tool

#load word2vec model
vec_model_file='./word2vect/finance/finance_vect_2.txt'
extend_vec_model_file='./word2vect/extend_vect_file.txt'
word_to_vector = word2vec_tool.MyWord2vec(vec_model_file,extend_vec_model_file)
word_to_vector.load()
batch_size=1
print('load word2vec model succ!')
#print(word_to_vector.max_word_len)
#print(len(word_to_vector.word_vect_dict))

def regular_embedding(embeddings):
    print(type(embeddings))
    print(type(embeddings[0]))
    print(embeddings[0])
    print(embeddings[len(embeddings)-1])
    result=[]
    for item in embeddings:
        result.append(item.tolist())
        #result.append(item)
    #print('shape embedding:',shape(result))
    return result 

def Seq2seq_dialog_train_test():
    #get Seq2seq_dialog_train model
    print('load Seq2seq model.....')
    checkpoint='./dialog_model.ckpt'
    #source_file='./data/ask_ori'
    #target_file='./data/answer_ori'
    source_file='./data/ask'
    target_file='./data/answer'
    encoder_rnn_size=100
    encoder_num_layers=2
    decoder_rnn_size=100
    decoder_num_layers=2
    epochs=100
    learn_rate=0.001
    word_to_vector_model=word_to_vector
    #special_embeddings=word_to_vector_model.random_vect()
    special_embeddings=regular_embedding(word_to_vector_model.extend_words_vect_matrix)
    general_embeddings=regular_embedding(word_to_vector_model.words_vect_matrix)
    train_model=Seq2seq_dialog_train.DialogTrain(checkpoint,source_file,target_file,\
        encoder_rnn_size,encoder_num_layers,decoder_rnn_size,decoder_num_layers,\
        epochs,batch_size,learn_rate,word_to_vector_model,special_embeddings,general_embeddings)

    #train
    print('train model....')
    target_int_to_word,target_word_to_int=train_model.train()
    #print(' model test....')
    #train_model.test()

def Seq2seq_dialog_train_predict_test():
    print('predict model....')
    checkpoint='./dialog_model.ckpt'
    target_int_to_word_file='./target_int_to_word.json'
    target_word_to_int_file='./target_word_to_int.json'
    predict_model=Seq2seq_dialog_predict.DialogPredict(checkpoint,\
        target_int_to_word_file,target_word_to_int_file,word_to_vector,batch_size)
    predict_model.predict('对')
    predict_model.predict('没问题')
    predict_model.predict('我一会就去存')
    predict_model.predict('没钱')
    predict_model.predict('不明白')
#def Seq2seq_dialog_predict_test():
#    #load word2vec model
#    vec_model_file='./word2vect/finance/finance_vect_2.txt'
#    word_to_vector = word2vec_tool.MyWord2vec(vec_model_file)
#    word_to_vector.load()
#    print('load word2vec model succ!')
#
#    checkpoint='./dialog_model.ckpt'
#    target_int_to_word_file='./target_int_to_word'
#    target_word_to_int_file='./target_word_to_int'
#    predict_model=Seq2seq_dialog_predict.DialogPredict(checkpoint\
#        target_int_to_word_file,target_word_to_int_file,word_to_vector)
#    predict_model.predict('是的')

if __name__=='__main__':
    Seq2seq_dialog_train_test()
    Seq2seq_dialog_train_predict_test()
