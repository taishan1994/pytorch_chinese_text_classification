# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F


"""这里就简单用全连接层实现"""
class FastText(nn.Module):
    def __init__(self, args, embedding_pretrained):
        super(FastText, self).__init__()
        self.embedding_pretrained = embedding_pretrained
        self.build_model(args)
        self.init_parameters()

    def build_model(self, args):
        if args.use_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(self.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(args.vocab_size, args.embedding_size, padding_idx=0)
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(args.embedding_size, args.hidden_size)
        self.maxpool = nn.MaxPool1d(args.max_seq_len)
        self.fc2 = nn.Linear(args.hidden_size, args.num_tags)

    def forward(self, x):

        out = self.embedding(x)
        out = self.dropout(out)
        out = self.fc1(out)
        out = F.relu(out)
        out = out.permute(0,2,1)
        out = self.maxpool(out).squeeze()
        out = self.fc2(out)
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