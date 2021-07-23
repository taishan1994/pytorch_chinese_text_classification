# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCnn(nn.Module):
    def __init__(self, args, embedding_pretrained=None):
        super(TextCnn, self).__init__()
        self.embedding_pretrained = embedding_pretrained
        self.build_model(args)
        self.init_parameters()

    def build_model(self, args):
        if args.use_pretrained:
            self.embedding = nn.Embedding.from_pretrained(self.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(args.vocab_size, args.embedding_size, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, args.num_filters, (k, args.embedding_size)) for k in args.filter_sizes])
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(args.num_filters * len(args.filter_sizes), args.num_tags)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        # x:[batchsize, max_length]
        out = self.embedding(x) # [batchsize, max_length, embedding_size]
        # unqueeze用于增加一个维度
        out = out.unsqueeze(1) # [batchsize, 1, max_length, embedding_size]
        # 这里分别使用不同size的卷积核进行卷积，以(2,, 300)为例
        # 输入：[64, 1, 32, 300]
        # 进行卷积得到输出：[64, 128, 32-2+1=31, 1]
        # 经过卷积之后经过一个relu，然后变形成[64, 128, 31]
        # 然后经过1维最大池化得[64, 128, 1]，再变形为[64, 128]
        # 最终将不同卷积核卷积后的结果拼接为,[64, 128*3]
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
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
