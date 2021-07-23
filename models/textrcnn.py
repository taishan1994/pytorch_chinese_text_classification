# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextRcnn(nn.Module):
    def __init__(self, args, embedding_pretrained):
        super(TextRcnn, self).__init__()
        self.embedding_pretrained = embedding_pretrained
        self.build_model(args)
        self.init_parameters()

    def build_model(self, args):
        if args.use_pretrained:
            self.embedding = nn.Embedding.from_pretrained(self.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(args.vocab_size, args.embedding_size, padding_idx=0)
        self.lstm = nn.LSTM(args.embedding_size, args.hidden_size, args.num_layers,
                            bidirectional=True, batch_first=True, dropout=args.dropout)
        self.maxpool = nn.MaxPool1d(args.max_seq_len)
        self.fc = nn.Linear(args.hidden_size * 2 + args.embedding_size, args.num_tags)

    def forward(self, x):
        embed = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        out, _ = self.lstm(embed)
        out = torch.cat((embed, out), 2) # [128, 32, 812]
        out = F.relu(out)
        out = out.permute(0, 2, 1) # torch.Size([128, 812, 32])
        print(out.shape)
        out = self.maxpool(out).squeeze() # [128, 812]
        print(out.shape)
        out = self.fc(out)
        print(out.shape)
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
