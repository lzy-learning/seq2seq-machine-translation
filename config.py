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
import os


class Config:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 训练集路径
        self.train_data_path = r'data/train_100k.jsonl'
        # 验证集路径
        self.valid_data_path = r'data/valid.jsonl'
        # 测试集路径
        self.test_data_path = r'data/test.jsonl'
        # 句子最大长度
        self.sentence_max_len = 50

        # 句子开始token和结束token
        self.SOS_token = 0
        self.EOS_token = 1
        # 句子填充由词表大小决定
        self.PAD_token = -1

        self.SOS_word = 'SOS'
        self.EOS_word = 'EOS'
        self.PAD_word = 'PAD'

        self.batch_size = 64
        self.hidden_size = 128

        self.lr = 0.001

        self.start_epoch = 0
        self.total_epoch = 80

        # 编码策略是否使用beam-search
        self.beam_search = True
        self.beam_width = 2

        # 使用teacher_forcing的概率有多大
        self.teacher_forcing_ratio = 1

        # 是否恢复训练
        self.is_resume = False
        self.resume_file = 'checkpoints/checkpoint.pth.tar'
        self.best_path = 'checkpoints/best.pth.tar'
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')

        # 日志记录
        self.logdir = 'log'
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)

        # 保存词典的路径，因为训练之后要用相同的词典对测试集操作
        self.input_lang_path = 'checkpoints/input_lang.pkl'
        self.output_lang_path = 'checkpoints/output_lang.pkl'


config = Config()
