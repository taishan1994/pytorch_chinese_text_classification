import numpy as np
import pickle
# 352217 300

vocab_size = 352221 + 2 # 总词量加上UNK# 和PAD
embedding_size = 300
vocab_path = '../cnews/final_data/wiki_char/vocab_char.txt'
char2embedding = {}
char2embedding_path = '../cnews/final_data/wiki_char/char2embedding.pkl'
embedding_path = '../cnews/final_data/wiki_char/wiki.char.embedding.pkl'

# PAD对应的向量
embedding_pad = [0 for _ in range(embedding_size)]
embedding_unk = list(np.random.normal(size=300))

embedding = np.zeros((vocab_size, embedding_size))
embedding[0, :] = embedding_pad
embedding[1, :] = embedding_unk

fp1 = open(vocab_path,'w',encoding='utf-8')
fp1.write('PAD' + '\n')
fp1.write('UNK' + '\n')
with open('sgns.wiki.char','r') as fp:
    lines = fp.read().strip().split('\n')
    for i,line in enumerate(lines):
        print(i, len(lines)-1)
        if i == 0:
            # 第一行是表示字数量以及嵌入维度
            continue
        line = line.strip().split(' ')
        word = line[0]
        fp1.write(word + '\n')
        vec = list(map(lambda x:float(x), line[1:]))
        embedding[i+1,:] = vec
        char2embedding[word] = vec
fp1.close()
with open(char2embedding_path,'wb') as fp:
    pickle.dump(char2embedding, fp)
with open(embedding_path,'wb') as fp:
    pickle.dump(embedding, fp)