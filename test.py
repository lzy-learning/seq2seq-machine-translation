import os
import pickle
import random
import torch
from torch import optim

from progress.bar import Bar
from config import config
from dataset.zh2enDataset import Zh2EnDataset, get_data, sentence2tensor, indices2sentence, index2word
from model.model import EncoderRNN, AttnDecoderRNN
from utils.utils import AverageMeter
from nltk.translate.bleu_score import sentence_bleu
import matplotlib.pyplot as plt
from matplotlib import ticker


def show_attention(input_sentence, output_words, attentions):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.cpu().numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels(input_sentence, rotation=90)
    ax.set_yticklabels(output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))


def test():
    # 需要注意Matplotlib绘图时可能不支持中文字符，需要进行特殊设置
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    if not os.path.exists(config.best_path):
        print('Train first!!')
        return

    # 读取句子对
    _, _, test_pairs = get_data(config.test_data_path)

    # 加载词典
    input_lang, output_lang = None, None
    with open(config.input_lang_path, 'rb') as fp:
        input_lang = pickle.load(fp)
    with open(config.output_lang_path, 'rb') as fp:
        output_lang = pickle.load(fp)

    # 加载模型、优化器参数
    encoder = EncoderRNN(input_lang.n_words, config.hidden_size).to(config.device)
    decoder = AttnDecoderRNN(config.hidden_size, output_lang.n_words).to(config.device)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=config.lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=config.lr)

    checkpoint = torch.load(config.resume_file)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
    decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])

    print('num of test sentence pairs: {}'.format(len(test_pairs)))

    # 开始测试
    bar = Bar(f"Testing......", max=len(test_pairs))
    test_bleu4 = AverageMeter()

    display_sentences = []

    with torch.no_grad():
        for test_input, test_output in test_pairs:
            test_tensor = sentence2tensor(input_lang, test_input)
            test_tensor = test_tensor.to(config.device)

            encoder_outputs, encoder_hidden = encoder(test_tensor)
            decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

            _, topi = decoder_outputs.topk(1)
            decoded_ids = topi.squeeze()

            decoded_words = []
            for idx in decoded_ids:
                if idx.item() == config.EOS_token:
                    break
                decoded_words.append(index2word(output_lang, idx.item()))

            # 这个函数默认计算bleu-4
            bleu4 = sentence_bleu([test_output], decoded_words)
            test_bleu4.update(bleu4, 1)

            # 展示四个句子
            if random.random() < 0.3 and len(display_sentences) < 4:
                display_sentences.append((test_input, test_output, decoded_words, decoder_attn))

            bar.suffix = 'BLEU-4 score: {:.4f}'.format(test_bleu4.avg)
            bar.next()
        bar.finish()

        for (src, target, predict, decoder_attn) in display_sentences:
            print('-----------------------------------')
            print('input sentence: {}'.format(''.join(src)))
            print('target sentence: {}'.format(' '.join(target)))
            print('predict sentence: {}'.format(' '.join(predict)))
            show_attention(src, predict, decoder_attn[0, :len(predict), :])

        print('=============>BLEU-4: {:.4f}<============='.format(test_bleu4.avg))
        plt.show()


if __name__ == '__main__':
    test()
