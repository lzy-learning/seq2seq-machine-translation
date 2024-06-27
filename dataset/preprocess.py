# E:\Program Files(x86)\pyProject\env python
# encoding: utf-8
'''
@author: LZY
@contact: 2635367587@qq.com
@file: preprocess.py
@time: 2024/6/13 9:27
@desc: 
'''
# coding=utf-8
import jieba
import unicodedata
# import sys, re, collections, nltk
# from nltk.stem.wordnet import WordNetLemmatizer
# from nltk.tokenize import word_tokenize

import re


def to_lower_case_mixed_text(mixed_text):
    '''
    将中文里的大写英文转为小写
    :param mixed_text:
    :return:
    '''

    # 定义一个函数，用于将匹配到的英文字符转换为小写
    def lower_case_match(match):
        return match.group(0).lower()

    # 使用正则表达式找到所有英文字符，并转换为小写
    lower_case_text = re.sub(r'[A-Za-z]', lower_case_match, mixed_text)
    return lower_case_text


def handle_acronym(text):
    """
    处理专有名词、缩写、特殊标点符号
    :param text:
    :return:
    """
    # acronym
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"cannot", "can not ", text)
    text = re.sub(r"what\'s", "what is", text)
    text = re.sub(r"What\'s", "what is", text)
    text = re.sub(r"\'ve ", " have ", text)
    text = re.sub(r"n\'t", " not ", text)
    text = re.sub(r"i\'m", "i am ", text)
    text = re.sub(r"I\'m", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r" e mail ", " email ", text)
    text = re.sub(r" e \- mail ", " email ", text)
    text = re.sub(r" e\-mail ", " email ", text)

    # spelling correction
    text = re.sub(r"ph\.d", "phd", text)
    text = re.sub(r"PhD", "phd", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" fb ", " facebook ", text)
    text = re.sub(r"facebooks", " facebook ", text)
    text = re.sub(r"facebooking", " facebook ", text)
    text = re.sub(r" usa ", " america ", text)
    text = re.sub(r" us ", " america ", text)
    text = re.sub(r" u s ", " america ", text)
    text = re.sub(r" U\.S\. ", " america ", text)
    text = re.sub(r" US ", " america ", text)
    text = re.sub(r" American ", " america ", text)
    text = re.sub(r" America ", " america ", text)
    text = re.sub(r" mbp ", " macbook-pro ", text)
    text = re.sub(r" mac ", " macbook ", text)
    text = re.sub(r"macbook pro", "macbook-pro", text)
    text = re.sub(r"macbook-pros", "macbook-pro", text)
    text = re.sub(r" 1 ", " one ", text)
    text = re.sub(r" 2 ", " two ", text)
    text = re.sub(r" 3 ", " three ", text)
    text = re.sub(r" 4 ", " four ", text)
    text = re.sub(r" 5 ", " five ", text)
    text = re.sub(r" 6 ", " six ", text)
    text = re.sub(r" 7 ", " seven ", text)
    text = re.sub(r" 8 ", " eight ", text)
    text = re.sub(r" 9 ", " nine ", text)
    text = re.sub(r"googling", " google ", text)
    text = re.sub(r"googled", " google ", text)
    text = re.sub(r"googleable", " google ", text)
    text = re.sub(r"googles", " google ", text)
    text = re.sub(r"dollars", " dollar ", text)

    # punctuation
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"-", " - ", text)
    text = re.sub(r"/", " / ", text)
    text = re.sub(r"\\", " \ ", text)
    text = re.sub(r"=", " = ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r"\.", " . ", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"\?", " ? ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\"", " \" ", text)
    text = re.sub(r"&", " & ", text)
    text = re.sub(r"\|", " | ", text)
    text = re.sub(r";", " ; ", text)
    text = re.sub(r"\(", " ( ", text)
    text = re.sub(r"\)", " ( ", text)

    # symbol replacement
    text = re.sub(r"&", " and ", text)
    text = re.sub(r"\|", " or ", text)
    text = re.sub(r"=", " equal ", text)
    text = re.sub(r"\+", " plus ", text)
    text = re.sub(r"\$", " dollar ", text)

    # remove extra space
    text = ' '.join(text.split())

    return text


class Rule:
    # 正则表达式过滤特殊符号用空格符占位，双引号、单引号、句点、逗号
    pat_letter = re.compile(r'[^a-zA-Z \']+')  # 保留'
    # 还原常见缩写单词
    pat_is = re.compile("(it|he|she|that|this|there|here)(\'s)", re.I)
    pat_s = re.compile("([a-zA-Z])(\'s)")  # 处理类似于这样的缩写today’s
    pat_not = re.compile("([a-zA-Z])(n\'t)")  # not的缩写
    pat_would = re.compile("([a-zA-Z])(\'d)")  # would的缩写
    pat_will = re.compile("([a-zA-Z])(\'ll)")  # will的缩写
    pat_am = re.compile("([I|i])(\'m)")  # am的缩写
    pat_are = re.compile("([a-zA-Z])(\'re)")  # are的缩写
    pat_ve = re.compile("([a-zA-Z])(\'ve)")  # have的缩写


def replace_abbreviations(text):
    new_text = text
    new_text = Rule.pat_letter.sub(' ', new_text).strip().lower()
    new_text = Rule.pat_is.sub(r"\1 is", new_text)  # 其中\1是匹配到的第一个group
    new_text = Rule.pat_s.sub(r"\1 ", new_text)
    new_text = Rule.pat_not.sub(r"\1 not", new_text)
    new_text = Rule.pat_would.sub(r"\1 would", new_text)
    new_text = Rule.pat_will.sub(r"\1 will", new_text)
    new_text = Rule.pat_am.sub(r"\1 am", new_text)
    new_text = Rule.pat_are.sub(r"\1 are", new_text)
    new_text = Rule.pat_ve.sub(r"\1 have", new_text)
    new_text = new_text.replace('\'', ' ')
    return new_text


# pos和tag有相似的地方，通过tag获得pos
# def get_wordnet_pos(treebank_tag):
#     if treebank_tag.startswith('J'):
#         return nltk.corpus.wordnet.ADJ
#     elif treebank_tag.startswith('V'):
#         return nltk.corpus.wordnet.VERB
#     elif treebank_tag.startswith('N'):
#         return nltk.corpus.wordnet.NOUN
#     elif treebank_tag.startswith('R'):  # 以副词
#         return nltk.corpus.wordnet.ADV
#     else:
#         return ''
#
#
# def merge(words):
#     lmtzr = WordNetLemmatizer()
#     new_words = ''
#     words = nltk.pos_tag(word_tokenize(words))  # tag is like [('bigger', 'JJR')]
#     for word in words:
#         pos = get_wordnet_pos(word[1])
#         if pos:
#             # lemmatize()方法将word单词还原成pos词性的形式
#             word = lmtzr.lemmatize(word[0], pos)
#             new_words += ' ' + word
#         else:
#             new_words += ' ' + word[0]
#     return new_words
#

def clean_line(text):
    text = handle_acronym(text)
    text = replace_abbreviations(text)
    # text = merge(text)
    text = text.strip()

    return text


if __name__ == '__main__':
    text = 'There\'re many recen\'t \'t extensions of this basic had idea to include attention. 120,had'
    text = clean_line(text)
    li = text.split()
    print(li)
    print(text)  # there are many rece not t extension of this basic have idea to include attention have
