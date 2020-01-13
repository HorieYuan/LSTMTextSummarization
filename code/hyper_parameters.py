UNK = '<unk>'
PAD = '<pad>'
BOS = '<bos>'
EOS = '<eos>'


# corpus
# 文本长
MAX_TEXT_LEN = 1000
# 句子最大长
MAX_SENTENCE_LEN = 30
# 文章句子数
MAX_SENTENCE_NUM = 150
# 摘要最大长
MAX_ABSTR_LEN = 150
# 摘要中的句子数
MAX_ABSTR_SENT_NUM = 20
# 词表大小
VOCAB_SIZE = 80000


# model
N_EPOCHS = 50
BATCH_SIZE = 64

EMBED_SIZE = 300

WORD_HIDDEN_SIZE = 100
WORD_NLAYERS = 2

SENTENCE_HIDDEN_SIZE = 200
SENTENCE_NLAYERS = 2

NDOC_DIMS = 500

LR = 0.0003


# path
VOCAB_PATH = '../data/vocab.pkl'

SENT_RNN_MODEL_PATH = '../model/sent_rnn.params'
ENCODER_MODEL_PATH = '../model/encoder.params'


# api
PORT = 10001
