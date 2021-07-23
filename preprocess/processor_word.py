import sys
sys.path.append('..')
import os
import logging
from configs import textcnn_config
from utils import utils
import jieba

logger = logging.getLogger(__name__)


class InputExample:
    def __init__(self, set_type, text, labels=None):
        self.set_type = set_type
        self.text = text
        self.labels = labels


class WordFeature:
    def __init__(self, word_ids, labels=None):

        self.word_ids = word_ids
        self.labels = labels


class Processor:

    @staticmethod
    def read_txt(file_path):
        with open(file_path, encoding='utf-8') as f:
            raw_examples = f.read().strip()
        return raw_examples

    def get_examples(self, raw_examples, set_type):
        examples = []
        # 这里是从json数据中的字典中获取
        for line in raw_examples.split('\n'):
            line = line.split('\t')
            examples.append(InputExample(set_type=set_type,
                                         text=line[0],
                                         labels=int(line[1])))
        return examples


def convert_word_example(ex_idx, example: InputExample, tokenizer, max_seq_len, word2id):
    set_type = example.set_type
    raw_text = example.text
    labels = example.labels
    # 文本元组
    callback_info = (raw_text,)
    callback_labels = labels
    callback_info += (callback_labels,)

    label_ids = labels
    # 分词
    word_list = tokenizer(raw_text, cut_all=False)
    word_ids = [word2id.get(word, 1) for word in word_list]
    if len(word_ids) < max_seq_len:
        word_ids = word_ids + [0] * (max_seq_len - len(word_ids))
    else:
        word_ids = word_ids[:max_seq_len]

    if ex_idx < 3:

        logger.info(f"*** {set_type}_example-{ex_idx} ***")
        logger.info(f"text: {word_list}")
        logger.info(f"word_ids: {word_ids}")
        logger.info(f"labels: {label_ids}")

    feature = WordFeature(
        word_ids=word_ids,
        labels=label_ids,
    )

    return feature, callback_info

def convert_examples_to_features(examples, max_seq_len, word2id):
    tokenizer = jieba.lcut
    features = []
    callback_info = []

    logger.info(f'Convert {len(examples)} examples to features')

    for i, example in enumerate(examples):
        feature, tmp_callback = convert_word_example(
            ex_idx=i,
            example=example,
            max_seq_len=max_seq_len,
            tokenizer=tokenizer,
            word2id=word2id,
        )
        if feature is None:
            continue

        features.append(feature)
        callback_info.append(tmp_callback)
    logger.info(f'Build {len(features)} features')

    out = (features,)

    if not len(callback_info):
        return out

    out += (callback_info,)
    return out

def get_out(processor, json_path, args, word2id, mode):
    raw_examples = processor.read_txt(json_path)

    examples = processor.get_examples(raw_examples, mode)
    for i, example in enumerate(examples):
        print(example.text)
        print(example.labels)
        if i == 5:
            break
    out = convert_examples_to_features(examples, args.max_seq_len, word2id)
    return out


if __name__ == '__main__':
    args = textcnn_config.Args().get_parser()
    args.log_dir = '../logs/'
    args.max_seq_len = 32
    args.data_dir = '../data/cnews/final_data/wiki_word/'
    utils.set_logger(os.path.join(args.log_dir, 'preprocess_word.log'))
    logger.info(vars(args))

    processor = Processor()

    label2id = {}
    id2label = {}
    with open('../data/cnews/final_data/wiki_word/labels.txt','r') as fp:
        labels = fp.read().strip().split('\n')
    for i,label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label
    print(label2id)

    word2id = {}
    id2word = {}
    with open('../data/cnews/final_data/wiki_word/vocab.txt','r') as fp:
        words = fp.read().strip().split('\n')
    for i,word in enumerate(words):
        word2id[word] = i
        id2word[i] = word
    # print(word2id)

    train_out = get_out(processor, '../data/cnews/raw_data/train.txt', args, word2id, 'train')
    dev_out = get_out(processor, '../data/cnews/raw_data/dev.txt', args, word2id, 'dev')
    test_out = get_out(processor, '../data/cnews/raw_data/test.txt', args, word2id, 'test')