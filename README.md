# TextSummarization

## 数据准备
利用 `code/utils.py` 对原始数据进行处理，输入为原始数据，输出为处理后为原文句子打上标签之后的数据。

原始数据每行为一条以制表符分割的“原文摘要对”。示例 `data/rawdata.sample`。

输出结果示例 `data/prepared_data.sample`。

```sh
cd code/
python utils.py -i ../data/rawdata.sample -o ../data/prepared_data.sample
```

## 训练

训练时的输入数据为数据准备阶段的结果。训练完成之后会在 `model/` 文件夹产生模型参数文件。

```sh
cd code/
python train.py -i ../data/prepared_data.sample
```

## 测试

训练完成之后可以挑选结果较好的一轮模型参数进行测试。

```sh
cd code/
python summarize.py -s ../model/sent_rnn30.params -e ../model/encoder30.params -t ../data/test_data.txt
```
