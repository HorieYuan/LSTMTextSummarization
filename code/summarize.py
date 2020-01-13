import getopt
import pickle
import sys
from os import path

from mxnet import nd

import hyper_parameters as hp
from summarnner_model import Encoder, SentenceRecurrent
from utils import _prepare_predict_data, try_gpu

N_EPOCHS = hp.N_EPOCHS
BATCH_SIZE = hp.BATCH_SIZE

EMBED_SIZE = hp.EMBED_SIZE

WORD_HIDDEN_SIZE = hp.WORD_HIDDEN_SIZE
WORD_NLAYERS = hp.WORD_NLAYERS

SENTENCE_HIDDEN_SIZE = hp.SENTENCE_HIDDEN_SIZE
SENTENCE_NLAYERS = hp.SENTENCE_NLAYERS

NDOC_DIMS = hp.NDOC_DIMS
LR = hp.LR

VOCAB_PATH = hp.VOCAB_PATH

CTX = try_gpu()


def _summarize(encoder, sent_rnn, X, ctx):
    X = X.as_in_context(ctx)
    sentence_hidden, doc_encode = encoder(X)
    sentence_hidden = nd.transpose(sentence_hidden, axes=(1, 0, 2))

    previous = sentence_hidden[0]

    Y_h = []
    for sent_hidden in sentence_hidden:

        y_h = sent_rnn(sent_hidden, previous, doc_encode)
        Y_h.append(y_h)

        previous = previous + sent_hidden * y_h
    Y_h = nd.stack(*Y_h)
    Y_h = nd.transpose(nd.squeeze(Y_h, axis=-1), axes=(1, 0))
    Y_h = nd.round(Y_h)

    return Y_h


def summarize(encoder, sent_rnn, word_vocab, source):
    sentences, sent_idx = _prepare_predict_data(source, word_vocab)
    sent_idx = nd.expand_dims(sent_idx, axis=0)
    Y_h = _summarize(encoder, sent_rnn, sent_idx, CTX)[0]
    summa = []
    for idx, i_y in enumerate(Y_h):
        if (i_y.asscalar() == 1):
            summa.append(sentences[idx])

    summa = list(map(lambda x: ''.join(x).replace('<pad>', ''), summa))
    return summa


def main(sent_rnn_model, encoder_model, test_data):

    source = open(test_data, 'r').read()

    with open(VOCAB_PATH, 'rb') as fr:
        word_vocab = pickle.load(fr)

    NWORDS = len(word_vocab)
    encoder = Encoder(NWORDS, EMBED_SIZE, WORD_HIDDEN_SIZE, WORD_NLAYERS,
                      SENTENCE_HIDDEN_SIZE, SENTENCE_NLAYERS, NDOC_DIMS)
    sent_rnn = SentenceRecurrent(SENTENCE_HIDDEN_SIZE)

    encoder.load_parameters(encoder_model, ctx=CTX)
    sent_rnn.load_parameters(sent_rnn_model, ctx=CTX)

    summa = summarize(encoder, sent_rnn, word_vocab, source)

    print(summa)


if __name__ == "__main__":

    argv = sys.argv

    sent_rnn_model = ''
    encoder_model = ''
    test_data = ''
    try:
        help_info = 'usage: ' + argv[0] + ' -s <sent_rnn_model> -e <encoder_model> -t <test_data>'
        opts, _args = getopt.getopt(argv[1:], 'hs:e:t:', ['sent_rnn_model=', 'encoder_model=', 'test_data='])

    except getopt.GetoptError:
        print(help_info)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(help_info)
            sys.exit()
        elif opt in ('-s', '--sent_rnn_model'):
            sent_rnn_model = arg
        elif opt in ('-e', '--encoder_model'):
            encoder_model = arg
        elif opt in ('-t', '--test_data'):
            test_data = arg
    if sent_rnn_model == '' or encoder_model == '' or test_data == '':
        print(help_info)
        sys.exit(2)

    main(sent_rnn_model, encoder_model, test_data)
