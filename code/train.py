import getopt
import pickle
import sys

import gluonnlp as nlp
from mxnet import gluon, init

import hyper_parameters as hp
from train_helper import train
from utils import get_tag_file_dataset, load_tag_file_raw_data, try_gpu
from summarnner_model import Encoder, SentenceRecurrent

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


def main_train(data_path):

    data_text, data_tag = load_tag_file_raw_data(data_path)
    data_set, word_vocab = get_tag_file_dataset(data_text, data_tag)

    with open(VOCAB_PATH, 'wb') as fw:
        pickle.dump(word_vocab, fw)

    NWORDS = len(word_vocab)

    encoder = Encoder(NWORDS, EMBED_SIZE, WORD_HIDDEN_SIZE, WORD_NLAYERS,
                      SENTENCE_HIDDEN_SIZE, SENTENCE_NLAYERS, NDOC_DIMS)
    sent_rnn = SentenceRecurrent(SENTENCE_HIDDEN_SIZE)

    train(encoder, sent_rnn, data_set, LR, BATCH_SIZE, N_EPOCHS, word_vocab, CTX)


def main(argv):

    input_file = ''

    try:
        help_info = 'usage: ' + argv[0] + ' -i <input_file>'
        opts, _args = getopt.getopt(argv[1:], 'hi:', ['input_file='])

    except getopt.GetoptError:
        print(help_info)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(help_info)
            sys.exit()
        elif opt in ('-i', '--input_file'):
            input_file = arg

    if input_file == '':
        print(help_info)
        sys.exit(2)

    main_train(input_file)


if __name__ == "__main__":
    main(sys.argv)
