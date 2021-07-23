# pytorch_chinese_classification
基于pytorch的中文文本分类，包含fasttext、textcnn、textrnn、textrnn_att、textrnn_bc、transformer、dpcnn

# 准备
首先去<a href="https://github.com/Embedding/Chinese-Word-Vectors">Chinese-Word-Vectors</a>下载好预训练的词向量，这里使用的"Wikipedia_zh 中文维基百科"中的word，下载好之后放在data/pretrained下面<br>
在data/pretrained下面有process_pretrained_word.py，用于将词向量转换为我们所需要的格式，运行完之后会在data/cnews/final_data/wiki_word下面生成vocab.txt以及wiki.word.embedding.pkl

# 相关说明
--checkponits：保存训练好的模型<br>
--configs：各模型的配置<br>
--data：数据文件位置<br>
--|--cnew：数据集位置<br>
--|--|--raw_data：数据位置<br>
--|--|--final_data/wiki_word/：标签、词汇表、转换后的词嵌入<br>
--|--pretrained：原始预训练词向量保存位置<br>
--dataset：转换成DataSet<br>
--example：各模型运行的相关示例<br>
--logs：数据处理及运行时的日志<br>
--models：模型<br>
--preprocess：处理数据<br>
--utils：辅助的一些函数<br>

# 流程
整个流程的顺序是：<br>
1、preprocess<br>
2、dataset<br>
3、example<br>
首先在preprocess中会对数据进行处理（对句子进行分词，将分词后的单词利用词汇表映射成数字，将标签映射成数字），然后在dataset中将处理后的数据转换成输入给DataLoader所需的格式，最后在example文件夹下加载数据进行训练、验证、测试和预测<br>
特别说明：对于变长lstm，我们需要额外传入每条句子的长度，因此有额外的dataset_bc_word.py和processor_bc_word.py<br>
在example中，每一个文件对应一个模型的示例，在main.sh中是运行的语句，比如：
```python
python textcnn_main.py --pretrained_dir="../data/cnews/final_data/wiki_word/" --pretrained_name="wiki.word.embedding.pkl" --data_dir="../data/cnews/final_data/wiki_word/" --log_dir="../logs/" --output_dir="../checkpoints/" --num_tags=10 --seed=123 --gpu_ids="0" --max_seq_len=32 --lr=3e-5 --train_batch_size=128 --train_epochs=10 --eval_batch_size=128 --dropout=0.3 --use_pretrained --vocab_size=352217 --embedding_size=300 --num_filters=256 --filter_sizes="2,3,4"
```
在每一个xxx_main.py中，都包含了训练、验证、测试以及预测功能，可根据相应情况进行注释。

# 运行结果
在测试集上的表现：
| 模型      | loss | micro_f1 |
| ----------- | ----------- | ----------- |
| textcnn      |  26.163740      | 0.8955 |
| textrnn   |      33.899487   |    0.8675    |
| textrnn_bc   |     25.980649    |  0.8975      |
| textrnn_att   |     27.346857    |   0.8865     |
| dpcnn   |      25.749150   |    0.8959    |
| transformer   |   29.856594      |   0.8861     |
| fasttext   |    31.577982     |    0.8709    |
| textrcnn   |     -    |     -   |

最后一个忘记测试了，=，=!
