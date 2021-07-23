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

    def forward(self, word_ids, seq_lens):
        # x:[batchsize, max_length]
        out = self.embedding(word_ids) # [batchsize, max_length, embedding_size]
        # seq_lens.shape:[128]
        _, idx_sort = torch.sort(seq_lens, dim=0, descending=True)  # 长度从长到短排序（index）
        # 这里要好好理解一下，举例说明：
        """
        假设seq_lens：tensor([13, 11, 12, 15, 14, 17, 16, 13, 14])
        data, idx_sort = torch.sort(seq_lens, dim=0, descending=True)
        输出：
        data：tensor([17, 16, 15, 14, 14, 13, 13, 12, 11])
        idx_sort：tensor([5, 6, 3, 4, 8, 0, 7, 2, 1])
        这里的idx_sort是按长度排序后，该值在原seq_lens中对应的索引，通过：
        torch.index_select(out, 0, idx_sort)这条语句，我们就可以对输入张量按句子真实长度排序。
        data2, idx_unsort = torch.sort(idx_sort)
        输出：
        data2：tensor([0, 1, 2, 3, 4, 5, 6, 7, 8])
        idx_unsort：tensor([5, 8, 7, 2, 3, 0, 1, 6, 4])
        最后我们要还原成原来张量的顺序，因为它是和标签一一对应的：我们要把13放回第0位，现在它在排序后的索引是5，将11放回第1位，现在它在排序后的索引是8，将12放回第二位，现在它在排序后的索引是7，以此类推就得到了：tensor([5, 8, 7, 2, 3, 0, 1, 6, 4])
        """
        _, idx_unsort = torch.sort(idx_sort)  # 排序后，原序列的 index
        out = torch.index_select(out, 0, idx_sort)
        seq_len = list(seq_lens[idx_sort])
        out = nn.utils.rnn.pack_padded_sequence(out, seq_len, batch_first=True)
        # [batche_size, seq_len, num_directions * hidden_size]
        out, (hn, _) = self.lstm(out)
        out = torch.cat((hn[2], hn[3]), -1)
        # out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        out = out.index_select(0, idx_unsort)
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
