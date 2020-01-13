import getopt
import glob
import itertools
import json
import multiprocessing
import re
import sys
import warnings
from itertools import chain
from os import path

import gluonnlp as nlp
import mxnet as mx
from mxnet import gluon, nd, test_utils
from nltk.translate.bleu_score import sentence_bleu

import hyper_parameters as hp

warnings.filterwarnings("ignore")

UNK = hp.UNK
PAD = hp.PAD
BOS = hp.BOS
EOS = hp.EOS

# 文本长
MAX_TEXT_LEN = hp.MAX_TEXT_LEN
# 句子最大长
MAX_SENTENCE_LEN = hp.MAX_SENTENCE_LEN
# 文章句子数
MAX_SENTENCE_NUM = hp.MAX_SENTENCE_NUM
# 摘要最大长
MAX_ABSTR_LEN = hp.MAX_ABSTR_LEN
# 摘要中的句子数
MAX_ABSTR_SENT_NUM = hp.MAX_ABSTR_SENT_NUM

VOCAB_SIZE = hp.VOCAB_SIZE

VOCAB = dict()
tokenizer = nlp.data.JiebaTokenizer()


def load_tag_file_raw_data(path='../data/tag_file.csv'):
    print('load_prepared_data', path)
    # 加载分好词、打好标签的tag_file
    data_text = []
    data_tag = []

    with open(path, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            sents = line.split('\t')
            tag = sents.pop()

            sents = list(map(lambda x: x.split(' '), sents))
            tag = list(map(int, tag.strip().split(',')))
            data_text.append(sents)
            data_tag.append(tag)

    return data_text, data_tag


def tag_one_hot(tag):
    """对 tag 进行 one_hot，结果为 MAX_SENTENCE_NUM 长

    Args:
        tag (str): '1,2,5'
    Returns:
        ndarray: [0, 1, 1, 0, 0, 1]
    """

    tag = nd.sum(nd.one_hot(nd.array(tag), MAX_SENTENCE_NUM), axis=0)
    return tag


def clip_pad_sentences(sentences):
    # pad句长
    sent_padder = nlp.data.PadSequence(MAX_SENTENCE_LEN, pad_val=PAD)
    # pad文章长
    sents_padder = nlp.data.PadSequence(MAX_SENTENCE_NUM, pad_val=[PAD] * MAX_SENTENCE_LEN)

    sents = list(map(sent_padder, sentences))
    sents = sents_padder(sents)

    return sents


def _convert_text_to_idx(text, word_vocab):
    """把 [['词语', '词语'], ['词语', '词语']]
        转成 [[4, 5], [6, 7]]

    Args:
        sentences ([type]): [description]
        word_vocab ([type]): [description]

    Returns:
        list: 
    """

    # 对每句话
    def convert(x):
        return word_vocab[x]

    text_idx = []

    # 对每篇文章
    for t in text:
        text_idx.append(list(map(convert, t)))
    text_idx = nd.array(text_idx)
    return text_idx


def get_tag_file_dataset(data_text, data_tag):

    # clip pad
    text = list(map(clip_pad_sentences, data_text))
    tag = list(map(tag_one_hot, data_tag))

    tokens = []
    for i in itertools.chain.from_iterable(text):
        tokens += i
    token_counter = nlp.data.count_tokens(tokens)
    word_vocab = nlp.Vocab(token_counter, max_size=VOCAB_SIZE)

    # convert to idx
    text = _convert_text_to_idx(text, word_vocab)

    data_set = list(zip(text, tag))
    data_set = gluon.data.SimpleDataset(data_set)
    return data_set, word_vocab


def max_next_sent_num(selected_sent_nums, rest_sent_nums, text, title):
    """计算接下来那句话可以添加到摘要句
    Args:
        selected_sent_nums (set): 文章中已经标记为摘要句的句子下标集合
        rest_sent_nums (set): 还没标记为摘要句的句子下标集合
        text (list): 每句话已经分好词的句子列表 [['第一', '个', '句子'], ['第二', '个']]
        title (list): 每句话已经分好词的摘要列表
    Returns:
        int: 接下来可以添加到摘要句中的句子的下标
    """
    selected_nums = selected_sent_nums
    sent2 = list(chain.from_iterable(title))
    max_score = 0
    max_idx = None

    # 遍历每个没有添加进来的句子
    for sent_num in rest_sent_nums:
        sent1 = []
        # 将其添加进来
        selected_nums.add(sent_num)
        for num in sorted(selected_nums):
            sent1 += text[num]
        # 计算加进来之后的分数
        score = sentence_bleu([sent1], sent2)
        # 比较分数
        if score > max_score:
            _max_score = score
            max_idx = sent_num
        # 将这句移除
        selected_nums.remove(sent_num)
    return max_idx


def cut_sent(para):
    """分句

    Args:
        para (str): 文章
    Returns:
        list: 分好句的文章
    """

    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")


def load_raw_data(path):
    """加载语料 每行是一对 "原文\t摘要" 对，中间以"\t"分割
    Returns:
        tuple: (data_content, data_summary)
            data_content (list): 字符串列表
            data_summary (list): 字符串列表
    """
    data_content = []
    data_summary = []

    with open(path, 'r', encoding='utf-8') as fr:

        for line in fr.readlines():
            try:
                line = line.strip().split("\t")
                content = line[0]
                summary = line[1]
                data_content.append(content)
                data_summary.append(summary)
            except:
                pass

    return data_content, data_summary


def sentences_tag(text_title_pair):
    """对一条 文章摘要对 进行分句、分词之后，对文章中的句子打标签。

    Args:
        text_title_pair (tuple): 二元组（为了多进程处理，一个参数方便）
            pair[0] = text (str): 字符串 原始文章
            pair[1] = title (str): 字符串 原始摘要

    Returns:
        tuple: (text, selected_sent_nums)
            text (str): 字符串，句子之间制表符分割，词语之间空格分割
            selected_sent_nums (str): text中作为摘要的那些句子的标号，标号之间逗号分割
        example:
            ('这 是 一句 话 。   这 是 第二 句 话。   第三 句 。', '1,2')
    """

    text = text_title_pair[0]
    title = text_title_pair[1]

    text = cut_sent(text)
    title = cut_sent(title)

    text = [tokenizer(t) for t in text]
    title = [tokenizer(t) for t in title]

    n_sents = len(text)
    # 剩余的句子
    selected_sent_nums = set()
    # 选中的句子
    rest_sent_nums = set(range(n_sents))
    # 摘要总长度
    lenght = 0

    try:
        while lenght < 300 and len(rest_sent_nums) > 0:

            next_idx = max_next_sent_num(selected_sent_nums, rest_sent_nums, text, title)
            selected_sent_nums.add(next_idx)

            rest_sent_nums.remove(next_idx)
            lenght += len(text[next_idx])

        selected_sent_nums = sorted(list(selected_sent_nums))
    except:

        fe = open('../data/failed_items.txt', 'a')
        fe.write(str(text_title_pair).strip() + '\n')
        return None

    selected_sent_nums = list(map(str, selected_sent_nums))
    selected_sent_nums = ','.join(selected_sent_nums)

    text = [' '.join(t) for t in text]
    text = '\t'.join(text)

    return text, selected_sent_nums


def try_gpu():
    if len(test_utils.list_gpus()) > 0:
        return mx.gpu()
    else:
        return mx.cpu()


def _prepare_predict_data(source, word_vocab):
    source = source.replace('\n', '').replace('\t', '').replace(' ', '')

    sentences = cut_sent(source)
    sentences = [tokenizer(sent) for sent in sentences]

    sentences = clip_pad_sentences(sentences)
    sent_idx = _convert_text_to_idx(sentences, word_vocab)

    # sentences: ['词语词语', '词语词语']
    # sent_idx： [[4, 5], [6, 7]]
    return sentences, sent_idx


def prepare_data(source_path, target_path):
    """准备数据
    """
    text_list, summa_list = load_raw_data(source_path)

    data = zip(text_list, summa_list)

    pool = multiprocessing.Pool()
    res = pool.map(sentences_tag, data)

    print('finish. waitting to write')

    with open(target_path, 'w', encoding='utf-8') as fw:
        e_count = 0
        for line in res:
            if line is not None:
                fw.write(line[0] + '\t' + line[1] + '\n')
            else:
                e_count += 1
    print(e_count, 'item(s) skip')

    print('finish writing')


def main(argv):
    # 对应prepare_data()方法的两个参数
    input_path = ''
    output_file = ''

    try:
        help_info = 'usage: ' + argv[0] + ' -i <input_path> -o <output_file>'
        opts, _args = getopt.getopt(argv[1:], 'hi:o:', ['input_path=', 'output_file='])

    except getopt.GetoptError:
        print(help_info)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(help_info)
            sys.exit()
        elif opt in ('-i', '--input_path'):
            input_path = arg
        elif opt in ('-o', '--output_file'):
            output_file = arg
    if input_path == '' or output_file == '':
        print(help_info)
        sys.exit(2)
    prepare_data(input_path, output_file)


if __name__ == "__main__":
    main(sys.argv)
