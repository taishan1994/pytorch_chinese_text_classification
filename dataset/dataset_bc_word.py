import sys
sys.path.append('..')
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
# 这里要显示的引入BertFeature，不然会报错
from preprocess.processor_bc_word import WordFeature
from preprocess.processor_bc_word import get_out, Processor
from configs import textrnn_bc_config


class ClassificationDataset(Dataset):
    def __init__(self, features):
        self.nums = len(features)

        self.word_ids = [torch.tensor(example.word_ids).long() for example in features]
        self.seq_lens = [torch.tensor(example.seq_lens).long() for example in features]
        self.labels = [torch.tensor(example.labels).long() for example in features]

    def __len__(self):
        return self.nums

    def __getitem__(self, index):

        data = {
            'word_ids':self.word_ids[index],
            'seq_lens':self.seq_lens[index],
            'labels':self.labels[index],
        }

        return data

if __name__ == '__main__':
    args = textrnn_bc_config.Args().get_parser()
    args.log_dir = '../logs/'
    args.max_seq_len = 32

    processor = Processor()

    word2id = {}
    id2word = {}
    with open('../data/cnews/final_data/wiki_word/vocab.txt','r') as fp:
        words = fp.read().strip().split('\n')
    for i,word in enumerate(words):
        word2id[word] = i
        id2word[i] = word
    # print(word2id)

    train_out = get_out(processor, '../data/cnews/raw_data/train.txt', args, word2id, 'train')
    features, callback_info = train_out
    train_dataset = ClassificationDataset(features)
    for data in train_dataset:
        print(data['word_ids'].shape)
        print(data['seq_lens'])
        print(data['labels'])
        break
    args.train_batch_size = 128
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.train_batch_size,
                              sampler=train_sampler,
                              num_workers=2)

    for step,train_data in enumerate(train_loader):
        print(step)
        print(train_data['word_ids'].shape)
        print(train_data['seq_lens'].shape)
        print(train_data['seq_lens'])
        print(train_data['labels'].shape)
        break
