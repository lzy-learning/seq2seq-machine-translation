import jsonlines
import jieba
from io import open
import re
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, Dataset
from dataset.preprocess import clean_line

import gensim
from gensim.models.word2vec import Word2Vec
from dataset.preprocess import clean_line

from config import config


class Lang:
    def __init__(self, name):
        '''
        构建词典，name用于标识不同词典。本任务中只有zh和en两个词典
        :param name:
        '''
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {config.SOS_token: 'SOS', config.EOS_token: 'EOS'}
        self.n_words = 2  # 词的数量初始化为2

    def add_word(self, word):
        '''
        添加一个词
        :param word:
        :return:
        '''
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def add_sentence(self, sentence: list):
        '''
        要求传入的句子是经过分词器处理的词汇列表
        :param sentence:
        :return:
        '''
        for word in sentence:
            self.add_word(word)

    def filter_out(self):
        '''
        滤去低频词，重新编排词典
        :return:
        '''
        words_to_keep = {word: count for word, count in self.word2count.items() if count >= 2}

        # 新的词到索引映射
        new_word2index = {}
        # 新的索引到词映射
        new_index2word = {config.SOS_token: 'SOS', config.EOS_token: 'EOS'}
        # 新的词频统计
        new_word2count = {}

        # 重新分配索引
        self.n_words = 2  # 从2开始，因为SOS和EOS已经占用了前两个索引

        for word, count in words_to_keep.items():
            new_word2index[word] = self.n_words
            new_word2count[word] = count
            new_index2word[self.n_words] = word
            self.n_words += 1

        # 更新词典
        self.word2index = new_word2index
        self.word2count = new_word2count
        self.index2word = new_index2word


def read_langs(path):
    # 中译英
    zh_lang = Lang('zh')
    en_lang = Lang('en')

    # 中英文句子对，列表中存放元组或者list
    sentence_list = []

    with open(path, encoding='utf-8') as fp:
        for unit in jsonlines.Reader(fp):
            zh_str = unit['zh']
            en_str = unit['en']

            '''
            中文数据清洗
            '''
            # 数据清洗：去除空格
            zh_str = zh_str.replace(' ', '')
            # 数据清洗：保留数字、汉字、英文、常规句子符号
            pattern = re.compile('[^\u4e00-\u9fa5^,^.^!^a-z^A-Z^0-9]')
            line = re.sub(pattern, '', zh_str)
            zh_str = ''.join(line.split())
            # seg_list = jieba.cut(zh_str, cut_all=True)
            # print("Full Mode: " + "/ ".join(seg_list))  # 全模式
            # seg_list = jieba.cut(zh_str, cut_all=False)
            # print("Default Mode: " + "/ ".join(seg_list))  # 精确模式
            # seg_list = jieba.cut_for_search(zh_str)  # 搜索引擎模式
            # print(", ".join(seg_list))

            zh_seg_list = [word for word in jieba.cut(zh_str)]  # 默认是精确模式

            '''
            英文数据清洗
            '''
            en_str = clean_line(en_str)
            en_seg_list = en_str.split()

            '''
            保存数据
            '''
            # 添加句子
            zh_lang.add_sentence(zh_seg_list)
            en_lang.add_sentence(en_seg_list)
            # 保存中英文句子对
            sentence_list.append([zh_seg_list, en_seg_list])

    return zh_lang, en_lang, sentence_list


def filter_sentence(sentence_pairs):
    '''
    将句子长度过长或过短的句子对删除
    :param sentence_pairs:
    :return:
    '''
    result = []
    for pair in sentence_pairs:
        if len(pair[0]) >= config.sentence_max_len or len(pair[1]) >= config.sentence_max_len \
                or len(pair[0]) < 4 or len(pair[1]) < 4:
            continue
        result.append(pair)
    return result


def get_data(path=config.train_data_path):
    '''
    读取数据并返回对应给Lang对象(Language，可以将词转为索引或将索引转为词)，返回句子对
    :param path:
    :return:
    '''
    src_lang, target_lang, sentence_pairs = read_langs(path)
    print("Read %s sentence pairs" % len(sentence_pairs))
    sentence_pairs = filter_sentence(sentence_pairs)
    print("Remaining %s sentences after filtering" % len(sentence_pairs))
    print("Counted words:")
    print(src_lang.name, src_lang.n_words)
    print(target_lang.name, target_lang.n_words)
    return src_lang, target_lang, sentence_pairs


def sentence2indices(lang: Lang, sentence):
    '''
    句子转为索引向量
    :param lang: Lang实例，负责词到索引的映射
    :param sentence: 具体句子
    :return:
    '''
    return [lang.word2index[word] if word in lang.word2index else 0 for word in sentence]


def sentence2tensor(lang: Lang, sentence):
    '''
    句子转化为张量，实际上就是先转成索引在ToTensor
    :param lang:
    :param sentence:
    :return:
    '''
    indices = sentence2indices(lang, sentence)
    indices.append(config.EOS_token)
    return torch.tensor(indices, dtype=torch.long, device=config.device).view(1, -1)


def index2word(lang: Lang, idx):
    return lang.index2word[idx] if idx < lang.n_words else 'unknown'


def indices2sentence(lang: Lang, indices):
    '''
    索引转成句子，遇到EOS即停止
    :param lang:
    :param indices:
    :return:
    '''
    res = []
    for idx in indices:
        if idx == config.EOS_token:
            res.append('<EOS>')
            break
        res.append(lang.index2word[idx] if idx < lang.n_words else '')
    return res


def pair2tensors(lang0: Lang, lang1: Lang, pair):
    '''
    句子对转成张量对
    :param lang0: 句子对中第一个句子对应的语言Lang
    :param lang1: 句子对中第二个句子对应的语言Lang
    :param pair: 句子对
    :return:
    '''
    input_tensor = sentence2tensor(lang0, pair[0])
    target_tensor = sentence2tensor(lang1, pair[1])
    return input_tensor, target_tensor


# one-hot形式的词向量效果显然不好
class Zh2EnDataset(Dataset):
    def __init__(self, path):
        super(Zh2EnDataset, self).__init__()

        self.input_lang, self.output_lang, self.pairs = get_data(path)
        self.input_lang.filter_out()
        self.output_lang.filter_out()

        self.n_pairs = len(self.pairs)
        self.input_indices = np.zeros((self.n_pairs, config.sentence_max_len), dtype=np.int_)
        self.output_indices = np.zeros((self.n_pairs, config.sentence_max_len), dtype=np.int_)
        for idx, (inp, oup) in enumerate(self.pairs):
            inp_id = sentence2indices(self.input_lang, inp)
            oup_id = sentence2indices(self.output_lang, oup)
            # 填充End Of Sentence标识
            inp_id.append(config.EOS_token)
            oup_id.append(config.EOS_token)

            self.input_indices[idx, :len(inp_id)] = inp_id
            self.output_indices[idx, :len(oup_id)] = oup_id

    def __len__(self):
        return self.n_pairs

    def __getitem__(self, index):
        input_ids = torch.LongTensor(self.input_indices[index])
        output_ids = torch.LongTensor(self.output_indices[index])
        return input_ids, output_ids
