from torch import nn
import torch
import torch.nn.functional as F
import os
import numpy as np
from time import time
from torch.autograd import Variable
from gensim.models import Word2Vec
from sklearn.metrics import roc_auc_score
from gensim.models import FastText
from data_process import load_esim_data_and_labels,yield_esim_data_and_labels

word2id_file="./data/word2id.txt"
read_file = open(word2id_file, "r")
word2id={}
for i in read_file:
    i=i.strip().split()
    word2id[i[0]]=int(i[1])
read_file.close()


#参数配置
epcho=1
batch_size=256 
num_to_ev=400 # 训练多少批，在本地评测一次
vocab_size=len(word2id) # 词典大小
embedding_dim=256 # 词向量维度
t_max_len=22 #title的最大长度
q_max_len=11 #query的最大长度
lr=0.0001 #学习率



#加载验证集
val_data=load_esim_data_and_labels("/home/kesci/work/data/eval_data/19_eval.csv",word2id,q_max_len=q_max_len,t_max_len=t_max_len)


# 每个词拼接使用128维的w2v和128维的fast向量到256维的组合向量表示
ce = np.random.uniform(-1, 1, [vocab_size + 1,embedding_dim])
word2vec_model = Word2Vec.load("/home/kesci/word2vec.model")
fast_model = FastText.load("./data/fast_w2v.model")
ce[0] = np.zeros(embedding_dim)
for i in word2id:
    try:
        ce[word2id[i]] = np.concatenate((word2vec_model[i],fast_model[i]))
    except:
        print(i)
