# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextRnnAtt(nn.Module):
    def __init__(self, args, embedding_pretrained):
        super(TextRnnAtt, self).__init__()
        self.embedding_pretrained = embedding_pretrained
        self.build_model(args)
        self.init_parameters()

    def build_model(self, args):
        if args.use_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(self.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(args.vocab_size, args.embedding_size, padding_idx=0)
        self.lstm = nn.LSTM(args.embedding_size, args.hidden_size, args.num_layers,
                            bidirectional=True, batch_first=True, dropout=args.dropout)
        self.tanh1 = nn.Tanh()
        # self.u = nn.Parameter(torch.Tensor(args.hidden_size * 2, args.hidden_size * 2))
        self.w = nn.Parameter(torch.zeros(args.hidden_size * 2))
        self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(args.hidden_size * 2, args.hidden_size2)
        self.fc2 = nn.Linear(args.hidden_size2, args.num_tags)

    def forward(self, x):
        emb = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        H, _ = self.lstm(emb)  # [batch_size, seq_len, hidden_size * num_direction]=[128, 32, 256]

        M = self.tanh1(H)  # [128, 32, 256]
        # M = torch.tanh(torch.matmul(H, self.u))
        # [128,32,256]×[256] = [128, 32]
        # 这里的意思是将每个词嵌入用一个值表示。然后经过softmax生成每个词的注意力权重
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [128, 32, 1]
        out = H * alpha  # [128,32,256]×[128,32,1] = [128, 32, 256]
        out = torch.sum(out, 1)  # [128, 256]
        out = F.relu(out)
        out = self.fc1(out) # [128, 128]
        out = self.fc2(out)  # [128, 10]
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
