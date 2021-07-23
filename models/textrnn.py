# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextRnn(nn.Module):
    def __init__(self, args, embedding_pretrained=None):
        super(TextRnn, self).__init__()
        self.embedding_pretrained = embedding_pretrained
        self.build_model(args)
        self.init_parameters()

    def build_model(self, args):
        if args.use_pretrained:
            self.embedding = nn.Embedding.from_pretrained(self.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(args.vocab_size, args.embedding_size, padding_idx=0)
        # batch_first用于标识第一个维度是batchsize
        self.lstm = nn.LSTM(args.embedding_size, args.hidden_size, args.num_layers, bidirectional=True, batch_first=True, dropout=args.dropout)

        self.fc = nn.Linear(args.hidden_size * 2, args.num_tags)

    def forward(self, x):
        # x:[batchsize, max_length]
        out = self.embedding(x) # [batchsize, max_length, embedding_size]
        out, _ = self.lstm(out)
        out = self.fc(out[:,-1,:]) # 句子最后时刻的hidden state
        return out

    def init_parameters(self, method='xavier', exclude='embedding'):
        for name, w in self.named_parameters():
            if exclude not in name:
                if 'weight' in name:
                    if method == 'xavier':
                        nn.init.xavier_normal_(w)
                    elif method == 'kaiming':
                        nn.init.kaiming_normal_(w)
                    else:
                        nn.init.normal_(w)
                elif 'bias' in name:
                    nn.init.constant_(w, 0)
                else:
                    pass
