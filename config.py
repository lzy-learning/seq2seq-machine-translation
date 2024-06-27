# E:\Program Files(x86)\pyProject\env python
# encoding: utf-8
'''
@author: LZY
@contact: 2635367587@qq.com
@file: config.py
@time: 2024/6/21 14:07
@desc: 
'''

import torch


class Config:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 训练集路径
        self.train_data_path = r'data/train_10k.jsonl'
        # 验证集路径
        self.valid_data_path = r'data/valid.jsonl'
        # 测试集路径
        self.test_data_path = r'data/test.jsonl'
        # 句子最大长度
        self.sentence_max_len = 50

        # 词向量长度由词表决定
        self.word_vector_len = 100

        # 句子开始token和结束token
        self.SOS_token = 0
        self.EOS_token = 1
        # 句子填充由词表大小决定
        self.PAD_token = -1

        self.batch_size = 64
        self.hidden_size = 128

        self.lr = 0.001

        self.start_epoch = 0
        self.total_epoch = 128

        # 使用teacher_forcing的概率有多大
        self.teacher_forcing_ratio = 1


config = Config()
