import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config


class EncoderRNN(nn.Module):
    '''
    编码器，对于输入的每个词向量，输出一个特征向量和一个隐状态
    '''

    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        # 隐层向量维度
        self.hidden_size = hidden_size
        # 词嵌入结构，如果使用预训练的Word2Vec模型，事实上Embedding可以不用
        self.embedding = nn.Embedding(input_size, hidden_size)
        # GRU结构
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        # query矩阵，将query将hidden_size维度映射到hidden_size
        self.Wa = nn.Linear(hidden_size, hidden_size)
        # key矩阵
        self.Ua = nn.Linear(hidden_size, hidden_size)
        # 获取总的分数
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        # 计算query和keys的加权和，通过tanh激活函数
        energy = torch.tanh(self.Wa(query) + self.Ua(keys))
        # 将加权和转换为分数
        scores = self.Va(energy)
        # 调整scores维度，使其适合后续的bmm操作
        scores = scores.squeeze(2).unsqueeze(1)
        # 使用softmax计算权重（对最后一个维度操作）
        weights = F.softmax(scores, dim=-1)
        # 使用batch matrix multiplication计算上下文向量并返回
        context = torch.bmm(weights, keys)

        return context, weights


class AttnDecoderRNN(nn.Module):
    '''
    普通的解码器只使用编码器的最后一个输出向量作为输入
    而基于attention的编码器
    '''

    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=config.device).fill_(config.SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        # 循环遍历每个时间步，直到达到句子的最大长度
        for i in range(config.sentence_max_len):
            # 调用 forward_step 方法来处理当前时间步
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            # 根据是否提供 target_tensor 决定使用教师强制还是自由运行策略
            if target_tensor is not None:
                # 如果提供了 target_tensor，使用教师强制策略
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                # Free running策略，使用解码器自身预测的 top-k 结果作为下一个输入
                _, topi = decoder_output.topk(1)
                # 将 top-k 的结果进行 squeeze 操作去除单维度，然后 detach 操作防止梯度的累积
                decoder_input = topi.squeeze(-1).detach()

        # 将所有时间步的解码器输出合并成一个张量
        decoder_outputs = torch.cat(decoder_outputs, dim=1)

        # 对合并后的解码器输出应用 log_softmax 函数，这通常用于多分类问题中，将输出转换为概率分布的对数形式
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)

        # 将所有时间步的注意力权重合并成一个张量
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions

    def forward_step(self, input, hidden, encoder_outputs):
        # 将输入的词向量转为连续的向量表示
        embedded = self.dropout(self.embedding(input))
        # 计算当前时间步的注意力权重，这些权重决定了编码器输出（通常是编码器的隐藏状态序列）在生成当前输出时的重要性
        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        # 使用注意力权重生成上下文向量，该向量是编码器输出的加权和，反映了输入序列中与当前输出最相关的信息
        input_gru = torch.cat((embedded, context), dim=2)
        # 循环神经网络单元GRU更新隐状态
        output, hidden = self.gru(input_gru, hidden)
        # 预测单词
        output = self.out(output)

        return output, hidden, attn_weights
