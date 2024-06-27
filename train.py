from io import open
import re
import unicodedata
import re
import random
import torch
import torch.nn as nn
from torch import optim

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from progress.bar import Bar
from config import config
from dataset.zh2enDataset import Zh2EnDataset, get_data, sentence2tensor, indices2sentence
from model.model import EncoderRNN, AttnDecoderRNN
from utils.utils import AverageMeter
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu


def train():
    '''
    进行中译英模型训练
    :return:
    '''

    '''
    准备训练数据
    '''
    train_dataset = Zh2EnDataset(config.train_data_path)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=config.batch_size,
        drop_last=False
    )

    _, _, valid_pairs = get_data(config.valid_data_path)

    '''
    准备模型、优化器、损失函数
    '''
    encoder = EncoderRNN(train_dataset.input_lang.n_words, config.hidden_size).to(config.device)
    decoder = AttnDecoderRNN(config.hidden_size, train_dataset.output_lang.n_words).to(config.device)

    # 优化器使用Adam
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=config.lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=config.lr)

    # 损失函数使用NLLLoss，即负对数自然损失：-log(p_y)
    criterion = nn.NLLLoss()

    '''
    开始训练
    '''
    for epoch in range(config.start_epoch, config.total_epoch):
        '''
        训练
        '''
        train_loss = AverageMeter()
        bar = Bar(f"[Epoch {epoch}/{config.total_epoch}]Train", max=len(train_dataloader))

        for batch_idx, (input_tensor, target_tensor) in enumerate(train_dataloader):
            input_tensor = input_tensor.to(config.device)
            target_tensor = target_tensor.to(config.device)
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            # encoder_outputs是编码器每个时期的输出，会用于Attention机制
            # encoder_hidden是编码最后一个时期的输出，保存了句子的整体特征信息
            encoder_outputs, encoder_hidden = encoder(input_tensor)
            # 决定是否使用teacher forcing
            if random.random() < config.teacher_forcing_ratio:
                decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)
            else:
                decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden)

            # 损失函数值
            loss = criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                target_tensor.view(-1)
            )
            loss.backward()

            # 梯度反向传播
            encoder_optimizer.step()
            decoder_optimizer.step()

            train_loss.update(loss.item(), input_tensor.size(0))

            bar.suffix = 'Batch: {}/{}; Train Loss: {:.4f}'.format(batch_idx, len(train_dataloader), train_loss.avg)
            bar.next()

        bar.finish()

        '''
        测试
        '''
        with torch.no_grad():
            bar = Bar(f"[Epoch {epoch}/{config.total_epoch}]Validate", max=len(valid_pairs))
            valid_bleu4 = AverageMeter()

            for valid_input, valid_output in valid_pairs:
                valid_input = sentence2tensor(train_dataset.input_lang, valid_input)
                valid_input = valid_input.to(config.device)

                encoder_outputs, encoder_hidden = encoder(valid_input)
                decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

                _, topi = decoder_outputs.topk(1)
                decoded_ids = topi.squeeze()

                decoded_words = []
                for idx in decoded_ids:
                    if idx.item() == config.EOS_token:
                        decoded_words.append('<EOS>')
                        break
                    decoded_words.append(train_dataset.output_lang.index2word[idx.item()])

                # 这个函数默认计算bleu-4
                bleu4 = sentence_bleu([valid_output], decoded_words)
                valid_bleu4.update(bleu4, 1)

                bar.suffix = 'BLEU-4 score: {:.4f}'.format(valid_bleu4.avg)
                bar.next()
            bar.finish()


if __name__ == '__main__':
    train()
