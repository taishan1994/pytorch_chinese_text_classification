import sys
sys.path.append('..')
import torch
from torch.utils.data import Dataset, DataLoader
# 这里要显示的引入BertFeature，不然会报错
from preprocess.processor_word import WordFeature
from preprocess.processor_word import get_out, Processor
from configs import textcnn_config


class ClassificationDataset(Dataset):
    def __init__(self, features):
        self.nums = len(features)

        self.word_ids = [torch.tensor(example.word_ids).long() for example in features]
        self.labels = [torch.tensor(example.labels).long() for example in features]

    def __len__(self):
        return self.nums

    def __getitem__(self, index):

        data = {
            'word_ids':self.word_ids[index],
            'labels':self.labels[index],
        }

        return data

if __name__ == '__main__':
    args = textcnn_config.Args().get_parser()
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
        print(data['labels'])
        break